from .agent import LogseqAgent
from .config import settings, configure_logging

# Ensure logging is configured when the package is imported
configure_logging()

__all__ = ["LogseqAgent", "settings"] 