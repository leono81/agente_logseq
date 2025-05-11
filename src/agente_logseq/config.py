# Configuración centralizada para AgenteLogseq y AnalysisAgent.
# Todas las variables se cargan automáticamente desde el entorno o .env gracias a pydantic-settings.
# Usar siempre 'from src.agente_logseq.config import settings' para acceder a la configuración.

from pydantic_settings import BaseSettings
import logfire

class Settings(BaseSettings):
    """
    Configuración centralizada para ambos agentes. No requiere load_dotenv ni os.getenv.
    Las variables se cargan automáticamente desde el entorno o un archivo .env en la raíz del proyecto.
    """
    OPENAI_API_KEY: str = ""
    LOGFIRE_SEND_TO_LOGFIRE: bool = False
    LOGFIRE_TOKEN: str | None = None
    LOGFIRE_CONSOLE_LOG_LEVEL: str = "INFO"
    LOGFIRE_CONSOLE_ENABLED: bool = True
    DEFAULT_LOGSEQ_GRAPH_PATH: str = ""

settings = Settings()

_logging_configured = False

def configure_logging():
    """
    Configura logfire para ambos agentes. Llamar solo una vez al inicio del programa.
    """
    global _logging_configured
    if _logging_configured:
        return

    effective_console_setting = False # Default to False (disabled)
    if settings.LOGFIRE_CONSOLE_ENABLED:
        effective_console_setting = logfire.ConsoleOptions(
            min_log_level=settings.LOGFIRE_CONSOLE_LOG_LEVEL.lower()
        )

    logfire.configure(
        send_to_logfire=settings.LOGFIRE_SEND_TO_LOGFIRE,
        token=settings.LOGFIRE_TOKEN,
        console=effective_console_setting, # Pass False or ConsoleOptions
        # pydantic_plugin=logfire.PydanticPlugin(log_validation_errors=True)
    )
    
    # Add Pydantic AI, OpenAI, and HTTPX instrumentation
    logfire.instrument_pydantic_ai()
    logfire.instrument_openai() 
    logfire.instrument_httpx()
    # logfire.instrument_requests() # requests is less likely to be used by modern openai sdk

    if not settings.OPENAI_API_KEY:
        logfire.error("OPENAI_API_KEY environment variable not set. The application may not function.")
    
    if settings.LOGFIRE_CONSOLE_ENABLED:
        logfire.info("Logfire configured with Pydantic AI, OpenAI, HTTPX instrumentation, and CONSOLE logging ENABLED.")
    else:
        logfire.info("Logfire configured with Pydantic AI, OpenAI, HTTPX instrumentation. CONSOLE logging DISABLED.")
        
    _logging_configured = True

# Este módulo debe ser la única fuente de configuración y logging para ambos agentes.

# Initial configuration call can be done here or explicitly in __init__.py or main.py
# configure_logging() # We will call this from __init__.py to ensure it runs on package import 