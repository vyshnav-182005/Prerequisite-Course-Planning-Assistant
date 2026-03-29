"""
Vector store module for the Course Planning Assistant.

Uses ChromaDB as a persistent vector store with Ollama embeddings
for embedding generation. Provides functions to build the index,
retrieve relevant chunks, clear the index, and get the total chunk count.
"""

import os
from typing import List

from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
TOP_K = int(os.getenv("TOP_K_RETRIEVAL", "5"))
COLLECTION_NAME = "course_catalog"

# ── Lazy-loaded singletons ────────────────────────────────────
_chroma_client = None
_collection = None
_embedding_function = None


def _get_embedding_function():
    """
    Get or create the Ollama embedding function.

    Returns:
        ChromaDB-compatible embedding function using Ollama embeddings.
    """
    global _embedding_function
    if _embedding_function is None:
        import requests

        class _OllamaEmbeddingFunction:
            def __init__(self, host: str, model: str) -> None:
                self._url = f"{host.rstrip('/')}/api/embeddings"
                self._model = model
                self._name = f"ollama:{model}"

            def name(self) -> str:
                return self._name

            def __call__(self, input: List[str]) -> List[List[float]]:
                embeddings: List[List[float]] = []
                for text in input:
                    # Handle both string and list inputs
                    if isinstance(text, list):
                        text = " ".join(str(t) for t in text)
                    
                    if not text or not str(text).strip():
                        embeddings.append([0.0] * 384)  # Default zero vector
                        continue
                    try:
                        payload = {"model": self._model, "prompt": str(text).strip()}
                        response = requests.post(self._url, json=payload, timeout=120)
                        response.raise_for_status()
                        data = response.json()
                        embedding = data.get("embedding")
                        if not embedding:
                            print(f"[Ollama Embedding] Warning: No embedding returned. Response: {data}")
                            embeddings.append([0.0] * 384)
                        else:
                            embeddings.append(embedding)
                    except requests.exceptions.RequestException as e:
                        print(f"[Ollama Embedding] Error embedding text: {e}")
                        print(f"[Ollama Embedding] Response text: {e.response.text if hasattr(e, 'response') else 'N/A'}")
                        embeddings.append([0.0] * 384)  # Fallback
                return embeddings

            def embed_documents(self, documents: List[str]) -> List[List[float]]:
                """Embed a list of documents."""
                return self(documents)

            def embed_query(self, input: str) -> List[float]:
                """Embed a single query string."""
                return self([input])[0]

        _embedding_function = _OllamaEmbeddingFunction(OLLAMA_HOST, EMBEDDING_MODEL)
    return _embedding_function


def _get_collection():
    """
    Get or create the ChromaDB collection.

    Uses persistent storage so embeddings survive across sessions.

    Returns:
        ChromaDB Collection object.
    """
    global _chroma_client, _collection
    if _collection is None:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_get_embedding_function(),
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def build_index(chunks: List[dict]) -> None:
    """
    Embed chunks and store in ChromaDB.

    Each chunk dictionary must have keys: content, chunk_id, source,
    section, and optionally page.

    Args:
        chunks: List of chunk dictionaries from the ingestion module.
    """
    if not chunks:
        print("[VectorStore] No chunks to index.")
        return

    collection = _get_collection()

    # Prepare batch data
    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "")
        ids.append(chunk_id)
        documents.append(chunk.get("content", ""))
        metadatas.append({
            "source": chunk.get("source", "unknown"),
            "section": chunk.get("section", "General"),
            "page": chunk.get("page") if chunk.get("page") is not None else -1,
        })

    # ChromaDB has a batch size limit; process in batches of 500
    batch_size = 500
    total_batches = (len(ids) + batch_size - 1) // batch_size

    print(f"[VectorStore] Indexing {len(ids)} chunks in {total_batches} batch(es)...")

    for i in range(0, len(ids), batch_size):
        batch_end = min(i + batch_size, len(ids))
        try:
            collection.add(
                ids=ids[i:batch_end],
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )
        except Exception as e:
            print(f"[VectorStore] ✗ Error indexing batch {i // batch_size + 1}: {e}")
            raise

    print(f"[VectorStore] ✓ Successfully indexed {len(ids)} chunks.")


def retrieve(query: str, k: int = TOP_K) -> List[dict]:
    """
    Retrieve top-k most relevant chunks for a query using cosine similarity.

    Args:
        query: The search query string.
        k: Number of top results to return (default from env).

    Returns:
        List of chunk dictionaries with keys: chunk_id, source, section,
        content, page.
    """
    collection = _get_collection()

    count = collection.count()
    if count == 0:
        print("[VectorStore] ⚠ No documents indexed. Please upload catalog documents first.")
        return []

    # Clamp k to number of available documents
    effective_k = min(k, count)

    try:
        # Generate the query embedding directly and pass it as query_embeddings.
        # This avoids a ChromaDB version-compatibility issue where the library may
        # call embed_query() (returning List[float]) instead of __call__()
        # (returning List[List[float]]) for query_texts, causing ChromaDB to
        # receive a 1-D list and raise:
        #   "argument 'query_embeddings': 'float' object cannot be converted to 'Sequence'"
        embed_fn = _get_embedding_function()
        query_embeddings = embed_fn([query])  # always List[List[float]]
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=effective_k,
        )
    except Exception as e:
        print(f"[VectorStore] ✗ Retrieval error: {e}")
        return []

    chunks = []
    if results and results.get("ids") and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
            content = results["documents"][0][i] if results.get("documents") else ""
            page_val = metadata.get("page", -1)
            chunks.append({
                "chunk_id": doc_id,
                "source": metadata.get("source", "unknown"),
                "section": metadata.get("section", "General"),
                "content": content,
                "page": page_val if page_val != -1 else None,
            })

    return chunks


def clear_index() -> None:
    """
    Clear all documents from ChromaDB for a new session.

    Deletes the existing collection and recreates it empty.
    """
    global _collection, _chroma_client
    try:
        import chromadb
        if _chroma_client is None:
            _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        # Delete and recreate the collection
        try:
            _chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # Collection might not exist

        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_get_embedding_function(),
            metadata={"hnsw:space": "cosine"},
        )
        print("[VectorStore] ✓ Index cleared.")
    except Exception as e:
        print(f"[VectorStore] ✗ Error clearing index: {e}")
        raise


def get_chunk_count() -> int:
    """
    Return total number of indexed chunks.

    Returns:
        Integer count of documents in the collection.
    """
    try:
        collection = _get_collection()
        return collection.count()
    except Exception as e:
        print(f"[VectorStore] ✗ Error getting chunk count: {e}")
        return 0
