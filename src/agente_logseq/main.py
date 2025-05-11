import os
import sys
import logfire

# Ensure the src directory is in the Python path if running main.py directly
# This is more robust for running as `python src/agente_logseq/main.py`
if __name__ == "__main__" and os.path.basename(os.getcwd()) != "agente_logseq" and os.path.dirname(__file__).endswith("src/agente_logseq"):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.agente_logseq.agent import LogseqAgent
from src.agente_logseq.config import settings, configure_logging
from src.agente_logseq.analysis_agent import AnalysisAgent


def get_multiline_input(prompt: str) -> str:
    """Gets multi-line input from the user."""
    print(prompt)
    lines = []
    while True:
        try:
            line = input()
            if line == "---": # Defined stop sequence for page creation input
                break
            if not line and lines: # Empty line after some input signals end for general/search queries
                # This allows single-line queries to be submitted with just one Enter
                # and multi-line text blocks (if not using ---) to end with double Enter.
                break 
            lines.append(line)
        except EOFError: # Ctrl+D
            break
    return "\n".join(lines)

def is_search_query(text: str) -> bool:
    """Heuristic to determine if the input text is likely a search query."""
    text_lower = text.lower()
    # Keywords that might indicate a search query
    search_keywords = [
        "search for", "find", "look for", "what is", "what are", "who is", "who are",
        "tell me about", "que es", "que son", "quien es", "quienes son", "buscar", 
        "encuentra", "que sabes de", "que sabemos de"
    ]
    if any(keyword in text_lower for keyword in search_keywords):
        return True
    
    # If it's a relatively short, single-line input, it's more likely a query
    if '\n' not in text and len(text.split()) < 15:
        return True
        
    return False

def is_task_list_query(text: str) -> bool:
    """Heuristic to determine if the input text is likely a task listing query."""
    text_lower = text.lower()
    # Keywords that might indicate a task listing query
    task_keywords = [
        "task", "tasks", "tarea", "tareas", "todo", "todos", "pending", "pendientes",
        "completed", "completadas", "list tasks", "show tasks", "view tasks",
        "listar tareas", "mostrar tareas", "ver tareas", "que tengo que hacer",
        "muestrame las tareas", "dame mis tareas"
    ]
    if any(keyword in text_lower for keyword in task_keywords):
        return True
    return False

def is_task_update_query(text: str) -> bool:
    """Heurística para detectar si el texto es una orden de actualización de estado de tarea."""
    text_lower = text.lower()
    update_keywords = [
        "marcar como terminada", "marcar como completada", "marcar como hecho", "marcar como finalizada",
        "tarea terminada", "tarea completada", "tarea hecha", "tarea finalizada",
        "completé la tarea", "completada la tarea", "he terminado la tarea", "he finalizado la tarea",
        "cambiar estado de la tarea", "actualizar estado de la tarea",
        "mark as done", "mark as completed", "finish task", "complete task", "set task as done",
        # NUEVOS SINÓNIMOS PARA REABRIR/ABRIR/PENDIENTE
        "marcar como abierta", "marcar como pendiente", "reabrir la tarea", "abrir la tarea",
        "poner como pendiente", "set as open", "set as pending", "reopen the task", "mark as open"
    ]
    return any(keyword in text_lower for keyword in update_keywords)

def is_logbook_query(text: str) -> bool:
    text_lower = text.lower()
    return (
        text_lower.startswith("logbook de tarea") or
        text_lower.startswith("historial de tarea") or
        text_lower.startswith("ver logbook de tarea") or
        text_lower.startswith("ver historial de tarea")
    )

def is_advanced_task_search_query(text: str) -> bool:
    text_lower = text.lower()
    return (
        text_lower.startswith("buscar tareas con") or
        text_lower.startswith("buscar tareas que contengan") or
        text_lower.startswith("buscar tareas por etiqueta")
    )

def is_import_plan_query(text: str) -> bool:
    text_lower = text.lower()
    return text_lower.startswith("importar plan ")

def is_create_graph_query(text: str) -> bool:
    """Detecta variantes de 'crear el grafo de:', 'agregar grafo:', 'añadir grafo:' al inicio del input."""
    import re
    return bool(re.match(r"^\s*(crear|agregar|añadir)(\s+el)?(\s+un)?\s*grafo(\s+de)?\s*:\s*", text, re.IGNORECASE))

def extract_graph_text(text: str) -> str:
    import re
    m = re.match(r"^\s*(crear|agregar|añadir)(\s+el)?(\s+un)?\s*grafo(\s+de)?\s*:\s*(.*)$", text, re.IGNORECASE | re.DOTALL)
    return (m.group(5).strip() if m and m.group(5) is not None else text.strip())

def is_create_plan_from_file_query(text: str) -> bool:
    """Detecta 'crear plan <ruta>' al inicio del input."""
    import re
    return bool(re.match(r"^\s*crear(\s+el)?(\s+un)?\s*plan\s+/.+", text, re.IGNORECASE))

def extract_plan_file_path(text: str) -> str:
    import re
    m = re.match(r"^\s*crear(\s+el)?(\s+un)?\s*plan\s+(/.+)$", text, re.IGNORECASE)
    return m.group(3).strip() if m else None

def is_create_task_query(text: str) -> bool:
    """Detecta si el input es una orden de crear tarea."""
    text_lower = text.lower()
    create_keywords = [
        "crear tarea", "nueva tarea", "agregar tarea", "añadir tarea", "add task", "new task", "create task"
    ]
    return any(keyword in text_lower for keyword in create_keywords)

def run_agent_cli():
    # Logging should be configured by the time LogseqAgent or settings are imported
    # via src.agente_logseq.__init__.py, but an explicit call here can ensure it
    # if running main.py as a script directly in some edge cases before package resolution.
    configure_logging() 

    if not settings.OPENAI_API_KEY:
        print("TARS > Critical error: OPENAI_API_KEY environment variable is not set. Deactivating.")
        logfire.error("OPENAI_API_KEY not set at CLI startup.")
        sys.exit(1)

    if not settings.DEFAULT_LOGSEQ_GRAPH_PATH or not os.path.isdir(settings.DEFAULT_LOGSEQ_GRAPH_PATH):
        print(f"TARS > Configuration error: DEFAULT_LOGSEQ_GRAPH_PATH ('{settings.DEFAULT_LOGSEQ_GRAPH_PATH}') is not set or not a valid directory. Deactivating.")
        sys.exit(1)

    logfire.info("TARS > Welcome. This is AgenteLogseq.")
    print("TARS > Welcome. This is AgenteLogseq.")
    agent = LogseqAgent(logseq_graph_path=settings.DEFAULT_LOGSEQ_GRAPH_PATH)
    logfire.info(f"Using Logseq graph directory: {settings.DEFAULT_LOGSEQ_GRAPH_PATH}")
    print(f"TARS > Logseq graph directory: {settings.DEFAULT_LOGSEQ_GRAPH_PATH}")

    print("TARS > Agent Logseq active. Ready for input.")
    print("TARS > - For page creation, provide detailed text and end with '---' on a new line.")
    print("TARS > - For search, type your query (e.g., 'search for my notes on X').")
    print("TARS > - Type 'exit' or 'quit' to deactivate.")

    while True:
        # Changed prompt to be more generic, actual mode (search/create) decided after input
        user_input_block = get_multiline_input("USER > ")

        if not user_input_block.strip(): # Handle empty input if user just presses enter twice
            print("TARS > Awaiting command.")
            continue

        if user_input_block.lower() in ["exit", "quit"]:
            print("TARS > Deactivating. It was a pleasure to serve.")
            break

        # --- NUEVO: crear grafo desde comando conversacional ---
        elif is_create_graph_query(user_input_block):
            print("TARS > Analizando texto y creando grafo. Stand by.")
            logfire.info("Flujo: crear grafo desde comando conversacional.")
            text = extract_graph_text(user_input_block)
            analyzer = AnalysisAgent()
            try:
                plan = analyzer.analyze_text_to_graph_plan(text)
                msg = agent.create_logseq_graph_from_plan(plan)
                print(f"TARS > {msg}")
            except Exception as e:
                logfire.error(f"Error en el flujo de crear grafo conversacional: {e}")
                print(f"TARS > Error al analizar o crear el grafo: {e}")
            continue

        # --- NUEVO: crear grafo desde archivo de texto ---
        elif is_create_plan_from_file_query(user_input_block):
            file_path = extract_plan_file_path(user_input_block)
            if not file_path or not os.path.isabs(file_path):
                print("TARS > Error: Debes proporcionar una ruta absoluta al archivo de texto.")
                continue
            if not os.path.isfile(file_path):
                print(f"TARS > Error: El archivo '{file_path}' no existe.")
                continue
            print(f"TARS > Analizando archivo '{file_path}' y creando grafo. Stand by.")
            logfire.info(f"Flujo: crear grafo desde archivo de texto: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            analyzer = AnalysisAgent()
            try:
                plan = analyzer.analyze_text_to_graph_plan(text)
                msg = agent.create_logseq_graph_from_plan(plan)
                print(f"TARS > {msg}")
            except Exception as e:
                logfire.error(f"Error en el flujo de crear grafo desde archivo: {e}")
                print(f"TARS > Error al analizar o crear el grafo: {e}")
            continue

        elif is_import_plan_query(user_input_block):
            print("TARS > Importando KnowledgeGraphPlan desde archivo. Stand by.")
            import re
            from src.models import KnowledgeGraphPlan
            m = re.match(r"importar plan (.+)", user_input_block.strip(), re.IGNORECASE)
            if not m:
                print("TARS > Error: Sintaxis incorrecta. Usa: importar plan <ruta>")
                continue
            plan_path = m.group(1).strip()
            if not os.path.isfile(plan_path):
                print(f"TARS > Error: El archivo '{plan_path}' no existe.")
                continue
            with open(plan_path, "r", encoding="utf-8") as f:
                plan_json = f.read()
            try:
                plan = KnowledgeGraphPlan.parse_raw(plan_json)
            except Exception as e:
                print(f"TARS > Error al procesar el plan: {e}")
                continue
            msg = agent.create_logseq_graph_from_plan(plan)
            print(f"TARS > {msg}")
            continue

        # --- NUEVO FLUJO: crear grafo desde texto libre ---
        elif user_input_block.strip().endswith('---'):
            print("TARS > Analizando texto y creando grafo. Stand by.")
            logfire.info("Flujo automático: análisis de texto y creación de grafo.")
            text = user_input_block.strip().removesuffix('---').strip()
            analyzer = AnalysisAgent()
            try:
                plan = analyzer.analyze_text_to_graph_plan(text)
                msg = agent.create_logseq_graph_from_plan(plan)
                print(f"TARS > {msg}")
            except Exception as e:
                logfire.error(f"Error en el flujo automático de grafo: {e}")
                print(f"TARS > Error al analizar o crear el grafo: {e}")
            continue

        # Nuevo orden de checks: Update de tarea, luego creación, luego listado, luego búsqueda avanzada, logbook, importar plan, búsqueda, luego creación
        if is_task_update_query(user_input_block):
            print("TARS > Interpretando como orden de actualización de tarea. Stand by.")
            logfire.info(f"User input identified as task update query: '{user_input_block}'")
            response = agent.handle_update_task_status_query(user_input_block)
        elif is_create_task_query(user_input_block):
            print("TARS > Interpretando como creación de tarea. Stand by.")
            logfire.info(f"User input identified as create task query: '{user_input_block}'")
            response = agent.handle_create_task_query(user_input_block)
        elif is_task_list_query(user_input_block):
            print("TARS > Interpreting as task list query. Stand by.")
            logfire.info(f"User input identified as task list query: '{user_input_block}'")
            response = agent.handle_get_all_tasks_query(user_input_block)
        elif is_advanced_task_search_query(user_input_block):
            print("TARS > Interpreting as advanced task search query. Stand by.")
            logfire.info(f"User input identified as advanced task search query: '{user_input_block}'")
            # Parse query, status, tag
            # Ejemplo: 'buscar tareas con voz', 'buscar tareas con voz y estado TODO', 'buscar tareas con voz y etiqueta PorHacer'
            import re
            query = ""
            status = None
            tag = None
            m = re.search(r"buscar tareas (?:que contengan|con) ([^\n]+)", user_input_block.lower())
            if m:
                query = m.group(1).strip()
            m2 = re.search(r"estado ([a-zA-Z]+)", user_input_block.lower())
            if m2:
                status = [m2.group(1).upper()]
            m3 = re.search(r"etiqueta ([^\s]+)", user_input_block.lower())
            if m3:
                tag = m3.group(1).strip().lstrip('#')
            result = agent.search_tasks(None, query=query, status_filter=status, tag_filter=tag)
            if not result.tasks:
                response = f"TARS > No se encontraron tareas para la búsqueda '{query}'."
            else:
                lines = [f"TARS > Tareas encontradas para '{query}':"]
                for t in result.tasks:
                    lines.append(f"- [{t.status}] {t.description} ({t.source_filename}, línea {t.line_number})")
                response = "\n".join(lines)
        elif is_logbook_query(user_input_block):
            print("TARS > Interpretando como consulta de logbook de tarea. Stand by.")
            logfire.info(f"User input identified as logbook query: '{user_input_block}'")
            response = agent.handle_task_logbook_query(user_input_block)
        elif is_search_query(user_input_block):
            print("TARS > Interpreting as search query. Stand by.")
            logfire.info(f"User input identified as search query: '{user_input_block}'")
            response = agent.handle_search_query(user_input_block)
        else:
            print("TARS > Interpreting as text for page creation. Analyzing... Stand by.")
            logfire.info(f"User input identified for page creation: '{user_input_block[:100]}...'")
            response = agent.process_text(user_input_block)
        
        print(f"TARS > {response}")

if __name__ == "__main__":
    run_agent_cli() 