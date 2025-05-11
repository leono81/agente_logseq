# Agente Logseq con Pydantic AI, OpenAI y RAG (ChromaDB)

## Descripción

Este agente permite interactuar con un grafo Logseq usando IA avanzada (Pydantic AI + OpenAI) y ahora soporta búsqueda semántica acelerada mediante RAG (Retrieval-Augmented Generation) usando ChromaDB y Sentence Transformers.

### Funcionalidades principales

- **Análisis y estructuración de texto:** Extrae temas y subtemas, crea páginas interconectadas en Logseq.
- **Creación de tareas:** Crea tareas (líneas TODO) en diarios o páginas específicas.
- **Búsqueda estructurada:** Busca páginas, diarios y tareas por texto, estado o etiqueta.
- **Actualización de tareas:** Cambia el estado de tareas existentes.
- **Consulta de logbook:** Muestra el historial de una tarea.
- **Búsqueda semántica (RAG):** Encuentra los bloques más relevantes del grafo usando embeddings y ChromaDB.
- **CLI interactiva:** Permite operar el agente desde terminal con comandos naturales.

## RAG: Búsqueda Semántica con ChromaDB

### ¿Qué es?
RAG permite buscar información relevante en todo el grafo Logseq usando similitud semántica, no solo coincidencia de palabras. Esto acelera y mejora la calidad de las respuestas del agente.

### ¿Cómo funciona?
- El indexador recorre todas las páginas y diarios, divide el contenido en bloques y los indexa en ChromaDB usando embeddings de Sentence Transformers.
- La herramienta `semantic_search_rag` permite buscar los bloques más relevantes para una consulta.
- El LLM puede usar estos resultados como contexto para respuestas más precisas.

### Comandos útiles

- **Reindexar el grafo:**
  ```python
  from src.rag.indexer import LogseqIndexer
  indexer = LogseqIndexer('/ruta/a/tu/grafo')
  indexer.index_logseq_graph(reindex=True)
  ```
- **Buscar semánticamente:**
  ```python
  from src.rag.search import LogseqSemanticSearcher
  searcher = LogseqSemanticSearcher()
  results = searcher.semantic_search('¿Cómo configuro HTTPS?', n_results=5)
  for hit in results:
      print(hit['document'], hit['metadata'])
  ```
- **Desde la CLI:**
  Puedes pedirle al agente que busque información relevante usando lenguaje natural, y el LLM usará la herramienta RAG si corresponde.

## Herramientas Pydantic AI expuestas
- `create_logseq_page`: Crea páginas con propiedades y enlaces.
- `search_logseq_graph`: Busca en páginas, diarios y tareas.
- `get_all_tasks`: Lista todas las tareas, con filtros.
- `update_task_status`: Cambia el estado de una tarea.
- `get_task_logbook`: Consulta el historial de una tarea.
- `search_tasks`: Busca tareas por texto, estado o etiqueta.
- `create_task`: Crea una nueva tarea (línea TODO).
- `semantic_search_rag`: Búsqueda semántica acelerada (RAG).

## Instalación

1. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Configura tu entorno (`.env` o variables de entorno):
   - `OPENAI_API_KEY`
   - `DEFAULT_LOGSEQ_GRAPH_PATH`

## Uso

- Ejecuta la CLI:
  ```bash
  python src/agente_logseq/main.py
  ```
- Sigue las instrucciones en pantalla para crear páginas, tareas, buscar información, etc.

## Notas
- El indexado RAG puede tomar tiempo la primera vez, pero acelera mucho las búsquedas posteriores.
- Puedes personalizar el modelo de embeddings en `src/rag/indexer.py` y `src/rag/search.py`.
- ChromaDB almacena el índice en `.chroma_index` por defecto.

## Notas importantes sobre RAG y ChromaDB

- El índice vectorial de ChromaDB ahora se guarda en una carpeta oculta `.chroma_index` dentro del directorio raíz de tu grafo Logseq.
- Esto asegura que el índice siempre esté junto a tus datos y evita confusiones con paths relativos.
- Si cambias de grafo, se creará un índice independiente para cada uno.

### Reindexar el grafo manualmente

Si necesitas forzar el reindexado (por ejemplo, tras agregar muchas páginas nuevas):

```python
from src.rag.indexer import LogseqIndexer
indexer = LogseqIndexer('/ruta/a/tu/grafo/logseq')
indexer.index_logseq_graph(reindex=True)
```

### Depuración desde la CLI

- `contar chunks rag`: muestra cuántos chunks hay indexados en el RAG.
- `ver chunks rag`: muestra los primeros 5 chunks indexados (texto y metadatos) para inspección rápida.

Si no ves chunks, asegúrate de que el path del grafo es correcto y que tienes páginas/diarios `.md` en tu Logseq.

---

**¡Contribuciones y feedback bienvenidos!** 