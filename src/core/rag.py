"""
RAG Orchestrator
Combines Retriever + Generator + Memory for end-to-end RAG pipeline
"""

from typing import Dict, Any, Optional, List
import time
from .retriever import HybridRetriever
from .generator import LLMGenerator
from config.settings import settings


class RAGPipeline:
    """
    Production-grade RAG orchestrator

    Features:
    - End-to-end RAG pipeline (retrieve + generate)
    - Conversation memory management
    - Comprehensive error handling
    - Performance monitoring
    - Health checks
    """

    def __init__(
        self,
        retriever_config: Optional[Dict[str, Any]] = None,
        generator_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize RAG pipeline

        Args:
            retriever_config: Optional config for retriever
            generator_config: Optional config for generator
        """
        # Initialize components
        self.retriever = HybridRetriever(**(retriever_config or {}))
        self.generator = LLMGenerator(**(generator_config or {}))

        # Status
        self._is_ready = False

    def initialize(self) -> bool:
        """
        Initialize all components (connect to services)

        Returns:
            bool: True if successful

        Raises:
            Exception: If initialization fails
        """
        try:
            print("=" * 80)
            print("  INITIALIZING RAG PIPELINE")
            print("=" * 80)
            print()

            # Connect retriever
            print("[1/2] Connecting retriever...")
            self.retriever.connect()
            print()

            # Connect generator
            print("[2/2] Connecting generator...")
            self.generator.connect()
            print()

            self._is_ready = True

            print("=" * 80)
            print("[SUCCESS] RAG Pipeline Ready!")
            print("=" * 80)
            print()

            return True

        except Exception as e:
            print(f"[ERROR] Failed to initialize RAG pipeline: {str(e)}")
            raise

    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
        use_memory: bool = True,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ask a question and get answer (full RAG pipeline)

        Args:
            question: User question
            top_k: Number of documents to retrieve (override default)
            use_memory: Whether to use conversation history (default: True)
            metadata_filter: Optional filter for retrieval (e.g., {"section_number": "5"})

        Returns:
            Dict with response and metadata:
                - answer: Generated response
                - sources: Retrieved documents
                - retrieval_time: Time taken for retrieval (seconds)
                - generation_time: Time taken for generation (seconds)
                - total_time: Total time (seconds)
                - success: Whether request succeeded

        Raises:
            RuntimeError: If pipeline not initialized
        """
        if not self._is_ready:
            raise RuntimeError("RAG pipeline not initialized. Call initialize() first.")

        # Validate input
        if not question or not question.strip():
            return {
                "answer": "Please provide a valid question.",
                "sources": [],
                "success": False,
                "error": "Empty question"
            }

        start_time = time.time()

        try:
            # Step 1: Retrieve relevant documents
            retrieval_start = time.time()
            results = self.retriever.retrieve(
                query=question,
                top_k=top_k,
                filter_dict=metadata_filter
            )
            retrieval_time = time.time() - retrieval_start

            # Step 2: Format context
            context = self.retriever.format_context(results)

            # Step 3: Generate response
            generation_start = time.time()
            response_data = self.generator.generate_with_fallback(
                question=question,
                context=context,
                use_memory=use_memory
            )
            generation_time = time.time() - generation_start

            # Calculate total time
            total_time = time.time() - start_time

            # Return comprehensive response
            return {
                "answer": response_data["response"],
                "sources": results,
                "retrieval_time": round(retrieval_time, 3),
                "generation_time": round(generation_time, 3),
                "total_time": round(total_time, 3),
                "num_sources": len(results),
                "success": response_data["success"],
                "warning": response_data.get("warning"),
                "error": response_data.get("error")
            }

        except Exception as e:
            print(f"[ERROR] RAG pipeline failed: {str(e)}")
            return {
                "answer": settings.ERROR_RESPONSE,
                "sources": [],
                "success": False,
                "error": str(e),
                "total_time": round(time.time() - start_time, 3)
            }

    def clear_conversation(self):
        """
        Clear conversation history
        """
        self.generator.clear_memory()

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history

        Returns:
            List of messages with role and content
        """
        return self.generator.memory.get_history()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics

        Returns:
            Dict with various stats
        """
        return {
            "retriever": self.retriever.health_check(),
            "generator": self.generator.health_check(),
            "ready": self._is_ready
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check

        Returns:
            Dict with health status of all components
        """
        status = {
            "pipeline_ready": self._is_ready,
            "retriever": self.retriever.health_check(),
            "generator": self.generator.health_check()
        }

        return status

    def close(self):
        """
        Cleanup all resources
        """
        print("[INFO] Closing RAG pipeline...")
        self.retriever.close()
        self.generator.close()
        self._is_ready = False
        print("[INFO] RAG pipeline closed")


# ========== Testing ==========

def test_rag_pipeline():
    """
    Test complete RAG pipeline
    """
    print("=" * 80)
    print("  TESTING RAG PIPELINE")
    print("=" * 80)
    print()

    try:
        # Initialize pipeline
        rag = RAGPipeline()
        rag.initialize()

        # Test questions
        test_questions = [
            "How many vacation days do I get per year?",
            "Can I carry over unused vacation days?",
            "What happens if I don't use all my vacation days?"
        ]

        # Ask questions
        for i, question in enumerate(test_questions, 1):
            print("\n" + "=" * 80)
            print(f"QUESTION {i}: {question}")
            print("=" * 80)

            result = rag.ask(question)

            print(f"\nAnswer: {result['answer']}")
            print(f"\nMetadata:")
            print(f"  Success: {result['success']}")
            print(f"  Sources: {result['num_sources']}")
            print(f"  Retrieval time: {result['retrieval_time']}s")
            print(f"  Generation time: {result['generation_time']}s")
            print(f"  Total time: {result['total_time']}s")

            if result.get('sources'):
                print(f"\nTop Source:")
                top_source = result['sources'][0]
                print(f"  Section: {top_source['section_number']} - {top_source['section_title']}")
                print(f"  Pages: {top_source['page_start']}-{top_source['page_end']}")
                print(f"  Score: {top_source['score']:.4f}")

        # Show conversation history
        print("\n" + "=" * 80)
        print("CONVERSATION HISTORY:")
        print("=" * 80)
        history = rag.get_conversation_history()
        for msg in history:
            role = msg['role'].upper()
            content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            print(f"{role}: {content}")

        # Health check
        print("\n" + "=" * 80)
        print("HEALTH CHECK:")
        print("=" * 80)
        health = rag.health_check()
        print(f"Pipeline Ready: {health['pipeline_ready']}")
        print(f"\nRetriever:")
        for key, value in health['retriever'].items():
            print(f"  {key}: {value}")
        print(f"\nGenerator:")
        for key, value in health['generator'].items():
            print(f"  {key}: {value}")

        print("\n[SUCCESS] RAG pipeline test completed!")

        # Cleanup
        rag.close()

    except Exception as e:
        print(f"\n[ERROR] Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    test_rag_pipeline()
