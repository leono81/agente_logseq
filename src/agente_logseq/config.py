import os
from dotenv import load_dotenv
import logfire

class Settings:
    OPENAI_API_KEY: str
    LOGFIRE_SEND_TO_LOGFIRE: bool
    LOGFIRE_TOKEN: str | None
    LOGFIRE_CONSOLE_LOG_LEVEL: str
    LOGFIRE_CONSOLE_ENABLED: bool
    DEFAULT_LOGSEQ_GRAPH_PATH: str

    def __init__(self):
        load_dotenv()
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.LOGFIRE_SEND_TO_LOGFIRE = os.getenv("LOGFIRE_SEND_TO_LOGFIRE", "false").lower() == "true"
        self.LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")
        self.LOGFIRE_CONSOLE_LOG_LEVEL = os.getenv("LOGFIRE_CONSOLE_LOG_LEVEL", "INFO").upper()
        self.LOGFIRE_CONSOLE_ENABLED = os.getenv("LOGFIRE_CONSOLE_ENABLED", "true").lower() == "true"
        self.DEFAULT_LOGSEQ_GRAPH_PATH = os.getenv("DEFAULT_LOGSEQ_GRAPH_PATH", "")

settings = Settings()

_logging_configured = False

def configure_logging():
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
        logfire.critical("OPENAI_API_KEY environment variable not set. The application may not function.")
    
    if settings.LOGFIRE_CONSOLE_ENABLED:
        logfire.info("Logfire configured with Pydantic AI, OpenAI, HTTPX instrumentation, and CONSOLE logging ENABLED.")
    else:
        logfire.info("Logfire configured with Pydantic AI, OpenAI, HTTPX instrumentation. CONSOLE logging DISABLED.")
        
    _logging_configured = True

# Initial configuration call can be done here or explicitly in __init__.py or main.py
# configure_logging() # We will call this from __init__.py to ensure it runs on package import 