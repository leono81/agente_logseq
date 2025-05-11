#!/usr/bin/env python3
"""
Script para indexar el grafo Logseq y crear el índice RAG (ChromaDB).
Uso:
    python src/rag/index_logseq_graph.py           # Usará el path de settings.DEFAULT_LOGSEQ_GRAPH_PATH
    python src/rag/index_logseq_graph.py /ruta/a/tu/grafo/logseq   # Override manual
"""
import sys
import os
from src.rag.indexer import LogseqIndexer
from src.agente_logseq.config import settings

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        graph_path = sys.argv[1]
    else:
        graph_path = settings.DEFAULT_LOGSEQ_GRAPH_PATH
    if not graph_path or not os.path.isdir(graph_path):
        print(f"Error: '{graph_path}' no es un directorio válido. (Configura DEFAULT_LOGSEQ_GRAPH_PATH o pásalo como argumento)")
        sys.exit(1)
    print(f"Indexando grafo Logseq en: {graph_path}")
    indexer = LogseqIndexer(graph_path)
    indexer.index_logseq_graph(reindex=True)
    print("Indexado RAG completo.")
    print(f"Ubicación del índice: {os.path.join(graph_path, '.chroma_index')}") 