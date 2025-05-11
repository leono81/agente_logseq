import os
import sys
import logfire

# Ensure the src directory is in the Python path if running main.py directly
# This is more robust for running as `python src/agente_logseq/main.py`
if __name__ == "__main__" and os.path.basename(os.getcwd()) != "agente_logseq" and os.path.dirname(__file__).endswith("src/agente_logseq"):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.agente_logseq.agent import LogseqAgent
from src.agente_logseq.config import settings, configure_logging


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

def run_agent_cli():
    # Logging should be configured by the time LogseqAgent or settings are imported
    # via src.agente_logseq.__init__.py, but an explicit call here can ensure it
    # if running main.py as a script directly in some edge cases before package resolution.
    configure_logging() 

    if not settings.OPENAI_API_KEY:
        print("TARS > Critical error: OPENAI_API_KEY environment variable is not set. Deactivating.")
        logfire.critical("OPENAI_API_KEY not set at CLI startup.")
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

        # Nuevo orden de checks: Update de tarea, luego listado, luego búsqueda, luego creación
        if is_task_update_query(user_input_block):
            print("TARS > Interpretando como orden de actualización de tarea. Stand by.")
            logfire.info(f"User input identified as task update query: '{user_input_block}'")
            response = agent.handle_update_task_status_query(user_input_block)
        elif is_task_list_query(user_input_block):
            print("TARS > Interpreting as task list query. Stand by.")
            logfire.info(f"User input identified as task list query: '{user_input_block}'")
            response = agent.handle_get_all_tasks_query(user_input_block)
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