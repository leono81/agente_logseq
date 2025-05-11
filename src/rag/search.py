import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import logfire
import os

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

class LogseqSemanticSearcher:
    def __init__(self, logseq_graph_path: str, chroma_path: str = None):
        if chroma_path is None:
            chroma_path = os.path.join(os.path.abspath(logseq_graph_path), ".chroma_index")
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection("logseq_chunks")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

    def semantic_search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        logfire.info(f"RAG: semantic_search ejecutada. Query: '{query}', n_results: {n_results}")
        query_embedding = self.embedder.encode([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        # results es un dict con keys: documents, metadatas, ids, distances
        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "id": results["ids"][0][i]
            })
        return hits 