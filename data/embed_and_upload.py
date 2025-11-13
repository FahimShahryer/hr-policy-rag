"""
Generate Dense + Sparse Embeddings and Upload to Pinecone
For Hybrid Search (Semantic + Keyword)
"""

import json
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from sentence_transformers import SentenceTransformer
import numpy as np

# Load environment
load_dotenv()


def load_chunks(chunks_file: str):
    """Load chunks from JSON file"""
    print(f"[1/6] Loading chunks from {chunks_file}")

    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    print(f"[OK] Loaded {len(chunks)} chunks")
    return chunks


def fit_bm25_encoder(chunks):
    """
    Fit BM25 encoder on entire corpus
    CRITICAL: Must fit on all documents for proper IDF statistics
    """
    print("\n[2/6] Fitting BM25 encoder on corpus...")

    # Extract all texts
    corpus = [chunk['text'] for chunk in chunks]

    # Initialize and fit BM25
    bm25_encoder = BM25Encoder.default()
    bm25_encoder.fit(corpus)

    print(f"[OK] BM25 encoder fitted on {len(corpus)} documents")
    return bm25_encoder


def generate_dense_embeddings(chunks):
    """
    Generate dense embeddings using sentence-transformers
    """
    print("\n[3/6] Generating dense embeddings...")
    print("Model: all-MiniLM-L6-v2 (384 dimensions)")

    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Extract texts
    texts = [chunk['text'] for chunk in chunks]

    # Batch encode all texts at once (efficient!)
    print(f"Encoding {len(texts)} texts...")
    dense_vectors = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    print(f"[OK] Generated {len(dense_vectors)} dense vectors")
    print(f"Vector shape: {dense_vectors.shape}")

    return dense_vectors


def generate_sparse_embeddings(chunks, bm25_encoder):
    """
    Generate sparse embeddings (BM25) using pinecone-text
    """
    print("\n[4/6] Generating sparse embeddings (BM25)...")

    # Extract texts
    texts = [chunk['text'] for chunk in chunks]

    # Encode documents (not queries!)
    sparse_vectors = bm25_encoder.encode_documents(texts)

    print(f"[OK] Generated {len(sparse_vectors)} sparse vectors")
    return sparse_vectors


def create_or_connect_index(pc, index_name: str, dimension: int):
    """
    Create Pinecone index or connect to existing one
    CRITICAL: Must use 'dotproduct' metric for hybrid search
    """
    print(f"\n[5/6] Setting up Pinecone index: {index_name}")

    # Check if index exists
    existing_indexes = pc.list_indexes().names()

    if index_name in existing_indexes:
        print(f"[OK] Index '{index_name}' already exists. Connecting...")
        index = pc.Index(index_name)

        # Show current stats
        stats = index.describe_index_stats()
        print(f"Current vectors: {stats.total_vector_count}")
    else:
        print(f"[INFO] Creating new index '{index_name}'...")
        print(f"  Dimension: {dimension}")
        print(f"  Metric: dotproduct (REQUIRED for hybrid search)")

        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='dotproduct',  # CRITICAL for hybrid search
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

        print(f"[OK] Index created. Connecting...")
        index = pc.Index(index_name)

    return index


def upload_to_pinecone(index, chunks, dense_vectors, sparse_vectors):
    """
    Upload dense + sparse vectors with metadata to Pinecone
    """
    print(f"\n[6/6] Uploading {len(chunks)} vectors to Pinecone...")

    # Prepare vectors for upload
    vectors_to_upsert = []

    for i, chunk in enumerate(chunks):
        # Prepare metadata (remove 'text' to save space, keep everything else)
        metadata = {
            'text': chunk['text'],  # Keep for retrieval
            'section_number': chunk['section_number'],
            'section_title': chunk['section_title'],
            'page_start': chunk['page_start'],
            'page_end': chunk['page_end'],
            'chunk_index': chunk['chunk_index'],
            'word_count': chunk['word_count'],
            'key_terms': chunk['key_terms'],
            'source_document': chunk['source_document']
        }

        # Create vector object with dense + sparse
        vector = {
            'id': chunk['id'],
            'values': dense_vectors[i].tolist(),  # Dense vector
            'sparse_values': sparse_vectors[i],   # Sparse vector (BM25)
            'metadata': metadata
        }

        vectors_to_upsert.append(vector)

    # Upload all vectors at once (103 chunks is small enough)
    print(f"Upserting {len(vectors_to_upsert)} vectors...")
    index.upsert(vectors=vectors_to_upsert)

    print(f"[OK] Upload complete!")

    # Verify upload
    print("\nVerifying upload...")
    stats = index.describe_index_stats()
    print(f"Total vectors in index: {stats.total_vector_count}")
    print(f"Index dimension: {stats.dimension}")

    return stats


def main():
    """
    Main pipeline: Embed and Upload for Hybrid Search
    """
    print("=" * 80)
    print("  EMBED AND UPLOAD TO PINECONE (HYBRID SEARCH)")
    print("=" * 80)
    print()

    # Configuration
    CHUNKS_FILE = "data/chunks.json"
    INDEX_NAME = "hr-policy-rag"
    DIMENSION = 384  # all-MiniLM-L6-v2 dimension

    # Load chunks
    chunks = load_chunks(CHUNKS_FILE)

    # Fit BM25 on corpus
    bm25_encoder = fit_bm25_encoder(chunks)

    # Generate dense embeddings
    dense_vectors = generate_dense_embeddings(chunks)

    # Generate sparse embeddings
    sparse_vectors = generate_sparse_embeddings(chunks, bm25_encoder)

    # Connect to Pinecone
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("[ERROR] PINECONE_API_KEY not found in .env")
        return

    pc = Pinecone(api_key=api_key)
    index = create_or_connect_index(pc, INDEX_NAME, DIMENSION)

    # Upload to Pinecone
    stats = upload_to_pinecone(index, chunks, dense_vectors, sparse_vectors)

    # Success summary
    print()
    print("=" * 80)
    print("  SUCCESS!")
    print("=" * 80)
    print()
    print(f"Index: {INDEX_NAME}")
    print(f"Total vectors: {stats.total_vector_count}")
    print(f"Dimension: {stats.dimension}")
    print()
    print("Hybrid Search Ready:")
    print("  - Dense vectors (semantic): sentence-transformers")
    print("  - Sparse vectors (keyword): BM25 via pinecone-text")
    print("  - Rich metadata: pages, sections, key terms")
    print()
    print("[NEXT STEP] Build RAG system with hybrid retrieval!")
    print("=" * 80)


if __name__ == "__main__":
    main()
