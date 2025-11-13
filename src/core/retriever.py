"""
Hybrid Retriever using Pinecone Native Hybrid Search
Combines dense (semantic) + sparse (BM25) search with server-side fusion
"""

import os
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from sentence_transformers import SentenceTransformer
import numpy as np
from config.settings import settings


class HybridRetriever:
    """
    Production-grade hybrid retriever with Pinecone native hybrid search

    Features:
    - Dense embeddings: sentence-transformers (semantic search)
    - Sparse embeddings: BM25 (keyword search)
    - Server-side fusion via Pinecone
    - Comprehensive error handling
    - Lazy loading of models
    """

    def __init__(
        self,
        index_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
        namespace: Optional[str] = None
    ):
        """
        Initialize hybrid retriever

        Args:
            index_name: Pinecone index name (default from settings)
            embedding_model: Sentence transformer model (default from settings)
            top_k: Number of results to retrieve (default from settings)
            alpha: Hybrid search balance (0=sparse, 1=dense, 0.5=balanced)
            namespace: Pinecone namespace (optional)
        """
        # Configuration
        self.index_name = index_name or settings.PINECONE_INDEX_NAME
        self.embedding_model_name = embedding_model or settings.EMBEDDING_MODEL
        self.top_k = top_k or settings.RETRIEVAL_TOP_K
        self.alpha = alpha or settings.RETRIEVAL_ALPHA
        self.namespace = namespace or settings.PINECONE_NAMESPACE

        # Models (lazy loaded)
        self._dense_model = None
        self._sparse_model = None
        self._pinecone_index = None

        # Connection status
        self._is_connected = False

    def connect(self) -> bool:
        """
        Connect to Pinecone and load models

        Returns:
            bool: True if successful

        Raises:
            Exception: If connection or model loading fails
        """
        try:
            # Initialize Pinecone
            if not settings.PINECONE_API_KEY:
                raise ValueError("PINECONE_API_KEY not set")

            pc = Pinecone(api_key=settings.PINECONE_API_KEY)

            # Check if index exists
            indexes = pc.list_indexes()
            if self.index_name not in indexes.names():
                raise ValueError(f"Index '{self.index_name}' does not exist")

            # Connect to index
            self._pinecone_index = pc.Index(self.index_name)

            # Verify index stats
            stats = self._pinecone_index.describe_index_stats()
            if stats.total_vector_count == 0:
                raise ValueError(f"Index '{self.index_name}' is empty")

            print(f"[OK] Connected to Pinecone index: {self.index_name}")
            print(f"[OK] Total vectors: {stats.total_vector_count}")

            # Load dense embedding model
            print(f"[INFO] Loading dense model: {self.embedding_model_name}")
            self._dense_model = SentenceTransformer(self.embedding_model_name)
            print(f"[OK] Dense model loaded")

            # Load sparse embedding model (BM25)
            print(f"[INFO] Loading BM25 encoder...")
            self._sparse_model = BM25Encoder.default()
            print(f"[OK] BM25 encoder loaded")

            self._is_connected = True
            return True

        except Exception as e:
            print(f"[ERROR] Failed to connect retriever: {str(e)}")
            raise

    def _generate_dense_embedding(self, text: str) -> List[float]:
        """
        Generate dense embedding for query

        Args:
            text: Query text

        Returns:
            List[float]: Dense embedding vector
        """
        if not self._dense_model:
            raise RuntimeError("Dense model not loaded. Call connect() first.")

        # Encode and normalize
        embedding = self._dense_model.encode(
            text,
            normalize_embeddings=True  # L2 normalization for dotproduct metric
        )

        return embedding.tolist()

    def _generate_sparse_embedding(self, text: str) -> Dict[str, Any]:
        """
        Generate sparse embedding (BM25) for query

        Args:
            text: Query text

        Returns:
            Dict with 'indices' and 'values' for sparse vector
        """
        if not self._sparse_model:
            raise RuntimeError("Sparse model not loaded. Call connect() first.")

        # Encode query (NOT document!)
        sparse_vector = self._sparse_model.encode_queries([text])[0]

        return sparse_vector

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using hybrid search

        Args:
            query: User question
            top_k: Number of results (override default)
            alpha: Hybrid balance (override default)
            filter_dict: Metadata filter (e.g., {"section_number": "5"})

        Returns:
            List of retrieved documents with metadata and scores

        Raises:
            RuntimeError: If retriever not connected
            Exception: If retrieval fails
        """
        if not self._is_connected:
            raise RuntimeError("Retriever not connected. Call connect() first.")

        # Use provided values or defaults
        k = top_k or self.top_k
        alpha_val = alpha or self.alpha

        try:
            # Generate embeddings
            dense_vec = self._generate_dense_embedding(query)
            sparse_vec = self._generate_sparse_embedding(query)

            # Query Pinecone with hybrid search
            # Pinecone handles server-side fusion automatically
            response = self._pinecone_index.query(
                vector=dense_vec,
                sparse_vector=sparse_vec,
                top_k=k,
                include_metadata=True,
                namespace=self.namespace,
                filter=filter_dict
            )

            # Parse results
            results = []
            for match in response.matches:
                result = {
                    "id": match.id,
                    "score": match.score,
                    "text": match.metadata.get("text", ""),
                    "section_number": match.metadata.get("section_number", ""),
                    "section_title": match.metadata.get("section_title", ""),
                    "page_start": match.metadata.get("page_start"),
                    "page_end": match.metadata.get("page_end"),
                    "key_terms": match.metadata.get("key_terms", []),
                    "source_document": match.metadata.get("source_document", ""),
                }
                results.append(result)

            return results

        except Exception as e:
            print(f"[ERROR] Retrieval failed: {str(e)}")
            raise

    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieved results into context string for LLM

        Args:
            results: List of retrieved documents

        Returns:
            str: Formatted context string
        """
        if not results:
            return "No relevant information found."

        context_parts = []
        for i, result in enumerate(results, 1):
            section = result.get("section_title", "Unknown Section")
            section_num = result.get("section_number", "")
            pages = f"Pages {result.get('page_start', '?')}-{result.get('page_end', '?')}"
            text = result.get("text", "").strip()

            context_part = f"""[Document {i}]
Section: {section_num} - {section}
{pages}
Content: {text}
"""
            context_parts.append(context_part)

        return "\n\n".join(context_parts)

    def health_check(self) -> Dict[str, Any]:
        """
        Check retriever health status

        Returns:
            Dict with health status information
        """
        status = {
            "connected": self._is_connected,
            "index_name": self.index_name,
            "dense_model": self.embedding_model_name,
            "sparse_model": "BM25Encoder",
            "top_k": self.top_k,
            "alpha": self.alpha,
        }

        if self._is_connected and self._pinecone_index:
            try:
                stats = self._pinecone_index.describe_index_stats()
                status["total_vectors"] = stats.total_vector_count
                status["dimension"] = stats.dimension
            except Exception as e:
                status["error"] = str(e)

        return status

    def close(self):
        """
        Cleanup resources
        """
        self._dense_model = None
        self._sparse_model = None
        self._pinecone_index = None
        self._is_connected = False
        print("[INFO] Retriever closed")


# ========== Testing ==========

def test_retriever():
    """
    Test retriever functionality
    """
    print("=" * 80)
    print("  TESTING HYBRID RETRIEVER")
    print("=" * 80)
    print()

    try:
        # Initialize and connect
        retriever = HybridRetriever()
        print("[1/3] Connecting to retriever...")
        retriever.connect()
        print()

        # Test query
        test_query = "What is the vacation policy?"
        print(f"[2/3] Testing query: '{test_query}'")
        results = retriever.retrieve(test_query)
        print(f"[OK] Retrieved {len(results)} results")
        print()

        # Show results
        print("[3/3] Results:")
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"    Score: {result['score']:.4f}")
            print(f"    Section: {result['section_number']} - {result['section_title']}")
            print(f"    Pages: {result['page_start']}-{result['page_end']}")
            print(f"    Text preview: {result['text'][:150]}...")

        # Format context
        print("\n" + "=" * 80)
        print("FORMATTED CONTEXT FOR LLM:")
        print("=" * 80)
        context = retriever.format_context(results[:3])
        print(context)

        # Health check
        print("\n" + "=" * 80)
        print("HEALTH CHECK:")
        print("=" * 80)
        health = retriever.health_check()
        for key, value in health.items():
            print(f"  {key}: {value}")

        print("\n[SUCCESS] Retriever test completed!")

    except Exception as e:
        print(f"\n[ERROR] Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    test_retriever()
