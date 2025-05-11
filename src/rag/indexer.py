import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logfire

# Puedes ajustar el modelo a uno más pequeño si quieres más velocidad
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

class LogseqIndexer:
    def __init__(self, logseq_graph_path: str, chroma_path: str = None):
        self.logseq_graph_path = os.path.abspath(logseq_graph_path)
        self.pages_path = os.path.join(self.logseq_graph_path, "pages")
        self.journals_path = os.path.join(self.logseq_graph_path, "journals")
        # Path absoluto y seguro para el índice
        if chroma_path is None:
            self.chroma_path = os.path.join(self.logseq_graph_path, ".chroma_index")
        else:
            self.chroma_path = chroma_path
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        self.collection = self.chroma_client.get_or_create_collection("logseq_chunks")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

    def index_logseq_graph(self, reindex: bool = False):
        """
        Indexa todos los bloques de páginas y diarios de Logseq en ChromaDB.
        Si reindex=True, borra la colección antes de indexar.
        """
        if reindex:
            logfire.info("RAG: Borrando colección previa de ChromaDB...")
            self.chroma_client.delete_collection("logseq_chunks")
            self.collection = self.chroma_client.get_or_create_collection("logseq_chunks")

        docs = []
        metadatas = []
        ids = []
        id_counter = 0
        total_chunks = 0

        # Indexar páginas
        for folder, ftype in [(self.pages_path, "page"), (self.journals_path, "journal")]:
            if not os.path.exists(folder):
                continue
            files = [fname for fname in os.listdir(folder) if fname.endswith(".md")]
            for fname in tqdm(files, desc=f"Indexando {ftype}s", unit="archivo"):
                abspath = os.path.join(folder, fname)
                with open(abspath, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        text = line.strip()
                        if not text: continue
                        chunk_id = f"{ftype}_{fname}_{i+1}"
                        docs.append(text)
                        metadatas.append({
                            "filename": fname,
                            "type": ftype,
                            "line": i+1,
                            "abspath": abspath
                        })
                        ids.append(chunk_id)
                        id_counter += 1
                        total_chunks += 1
                        # Para evitar chunks demasiado grandes en memoria
                        if len(docs) >= 128:
                            self._upsert_chunks(docs, metadatas, ids)
                            docs, metadatas, ids = [], [], []
        # Indexar los que queden
        if docs:
            self._upsert_chunks(docs, metadatas, ids)
        logfire.info(f"RAG: Indexado completo. Total de chunks indexados: {total_chunks}")
        print(f"\nRAG: Indexado completo. Total de chunks indexados: {total_chunks}")

    def _upsert_chunks(self, docs, metadatas, ids):
        embeddings = self.embedder.encode(docs, show_progress_bar=False)
        self.collection.upsert(
            documents=docs,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def count_indexed_chunks(self):
        return self.collection.count()

    def get_sample_chunks(self, n: int = 5):
        """
        Devuelve los primeros N chunks indexados (texto y metadatos) para depuración.
        """
        results = self.collection.get(limit=n, include=["documents", "metadatas", "ids"])
        chunks = []
        for i in range(len(results["ids"])):
            chunks.append({
                "id": results["ids"][i],
                "document": results["documents"][i],
                "metadata": results["metadatas"][i]
            })
        return chunks 