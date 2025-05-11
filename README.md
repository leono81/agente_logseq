# AgenteLogseq

An AI agent for interacting with Logseq, built with Python, Pydantic AI, and OpenAI.

## Description

AgenteLogseq analyzes textual input, identifies key concepts and relationships, 
and structures this information into interconnected Logseq pages. 
It aims for a formal and concise interaction style, inspired by TARS from Interstellar.

## Features

-   Receives blocks of text for analysis.
-   Uses an LLM (OpenAI's gpt-4o-mini via Pydantic AI) to:
    -   Identify main topics and sub-topics/entities.
    -   Determine relationships between them.
-   Generates and saves information in Logseq:
    -   Creates a main page for the central topic.
    -   Creates separate pages for sub-topics/entities.
    -   Interlinks pages using `[[Logseq Link Syntax]]`.
    -   Adds properties like `type:: concept`, `source:: chat_input`.
-   Logs operations using Logfire.

## Setup

1.  **Clone the repository (or create the files as shown).**

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the project root. You can copy the content below:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    LOGFIRE_SEND_TO_LOGFIRE="false" # Set to "true" to send logs to Logfire
    LOGFIRE_TOKEN="your_logfire_token_here" # Required if LOGFIRE_SEND_TO_LOGFIRE is true
    LOGFIRE_CONSOLE_LOG_LEVEL="INFO" # e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL
    DEFAULT_LOGSEQ_GRAPH_PATH="/path/to/your/logseq/graph" # Optional: set a default path
    ```
    Edit the `.env` file and fill in your details.

## Running the Agent

Execute the main script from the project root directory:
```bash
python -m src.agente_logseq.main
```

The agent will prompt you for the path to your Logseq graph directory (unless `DEFAULT_LOGSEQ_GRAPH_PATH` is set and valid in your `.env` file) and then await your text input.

## Project Structure

```
.
├── .env                # Your local environment variables (create this from example)
├── README.md
├── requirements.txt
└── src
    └── agente_logseq
        ├── __init__.py
        ├── agent.py
        ├── config.py
        └── main.py
```

## Tooling

-   **Pydantic AI**: For LLM interaction and tool integration.
-   **OpenAI API**: For language model capabilities.
-   **Logfire**: For logging and observability.
-   **Logseq**: Target knowledge base. 