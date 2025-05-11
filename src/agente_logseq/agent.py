import os
import re
from typing import Dict, List, Optional, Literal, Any, Union
import logfire

# Imports for pydantic-ai (version 0.1.11)
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.messages import SystemPromptPart, UserPromptPart
from pydantic import BaseModel, Field

from .config import settings

SearchableLocation = Literal["pages", "journals", "tasks"]

# --- Pydantic Models for Structured Search Results ---
class PageMatch(BaseModel):
    title: str
    filename: str
    path: str
    content_snippets: List[str] = Field(default_factory=list)
    tasks_found: List[str] = Field(default_factory=list)
    match_type: Literal["title", "content", "task"] = "content" 

class JournalMatch(BaseModel):
    date_str: str # e.g., "2024_11_25"
    filename: str
    path: str
    content_snippets: List[str] = Field(default_factory=list)
    tasks_found: List[str] = Field(default_factory=list)
    match_type: Literal["content", "task"] = "content"

class TaskMatch(BaseModel):
    description: str
    status: str # e.g., "TODO", "DONE"
    source_filename: str
    source_path: str
    source_type: Literal["page", "journal"]
    line_number: int
    # For journal tasks, we might infer date from filename later if needed
    source_page_title: Optional[str] = None 
    source_journal_date_str: Optional[str] = None

class StructuredSearchResults(BaseModel):
    query: str
    pages: List[PageMatch] = Field(default_factory=list)
    journals: List[JournalMatch] = Field(default_factory=list)
    # We can simplify by including tasks within PageMatch/JournalMatch for now, 
    # or have a separate top-level tasks list if tasks are truly distinct entities found.
    # For now, tasks_found in PageMatch/JournalMatch will store task strings.
    # If we want specific TaskMatch objects, the logic in search_logseq_graph would need more significant changes.
    # Let's stick to tasks_found: List[str] within Page/JournalMatch for simplicity first.

class StructuredTaskResults(BaseModel): # New model for get_all_tasks
    tasks: List[TaskMatch]
    filter_applied: Optional[List[str]] = None

class TaskLogbookEntry(BaseModel):
    clock_lines: list[str] = Field(default_factory=list)
    raw_logbook: str = ""

class TaskLogbookResult(BaseModel):
    description: str
    status: str
    source_filename: str
    line_number: int
    logbook: TaskLogbookEntry
    error: Optional[str] = None

class SearchTasksResult(BaseModel):
    tasks: list[TaskMatch]
    query: str
    status_filter: Optional[list[str]] = None
    tag_filter: Optional[str] = None

# --- End Pydantic Models ---

# Define a generic RunContext for tool type hinting
AnyRunContext = RunContext[Any]

class LogseqAgent:
    """
    AI Agent for interacting with Logseq, with a TARS-like personality.
    Analyzes text input and structures it into Logseq pages.
    """
    def __init__(self, logseq_graph_path: str):
        """
        Initializes the LogseqAgent.

        Args:
            logseq_graph_path: Path to the root of the Logseq graph.
        """
        self.logseq_graph_path = os.path.abspath(logseq_graph_path)
        self.logseq_pages_path = os.path.join(self.logseq_graph_path, "pages")
        self.logseq_journals_path = os.path.join(self.logseq_graph_path, "journals")
        
        logfire.info(
            f"LogseqAgent initialized. Graph path: {self.logseq_graph_path}, Pages: {self.logseq_pages_path}, Journals: {self.logseq_journals_path}"
        )

        if not settings.OPENAI_API_KEY:
            logfire.error("OpenAI API key not found in settings during agent initialization.")
            raise ValueError("OpenAI API key is required and not found in settings.")

        llm_model_instance = OpenAIModel(
            model_name="gpt-4o-mini"
        )

        agent_model_settings = {"temperature": 0.2}

        # Initialize the Agent first, tools will be registered via decorators/helper method.
        self.llm = Agent(
            model=llm_model_instance,
            model_settings=agent_model_settings
            # tools=[] # Tools are registered via @self.llm.tool or helper method
        )
        self._register_tools() # Register tools after llm is initialized

    def _register_tools(self):
        """Helper method to explicitly register tools with the LLM agent."""
        self.create_logseq_page = self.llm.tool(self.create_logseq_page)
        self.search_logseq_graph = self.llm.tool(self.search_logseq_graph)
        self.get_all_tasks = self.llm.tool(self.get_all_tasks)
        self.update_task_status = self.llm.tool(self.update_task_status)
        self.get_task_logbook = self.llm.tool(self.get_task_logbook)
        self.search_tasks = self.llm.tool(self.search_tasks)

    def _sanitize_filename(self, title: str) -> str:
        """
        Sanitizes a page title to be a valid filename for Logseq pages.
        - Replaces slashes with hyphens.
        - Removes other OS-problematic characters.
        - Strips leading/trailing whitespace and dots.
        """
        sanitized_title = title.replace('/', '-')
        sanitized_title = re.sub(r'[<>:"\\|?*]', '_', sanitized_title) # Escaped backslash
        sanitized_title = sanitized_title.strip(" .")
        if not sanitized_title:
            sanitized_title = "untitled_page"
        return sanitized_title

    # Tool registration will be handled by _register_tools
    def create_logseq_page(
        self,
        ctx: AnyRunContext, # Added ctx parameter for tool compatibility
        title: str,
        content_summary: str,
        properties: Optional[Dict[str, str]] = None,
        related_page_titles: Optional[List[str]] = None
    ) -> str:
        """
        Creates a new page file in the Logseq graph structure.
        The LLM calls this function with analyzed data to generate Logseq pages.

        Args:
            ctx: The Pydantic AI run context (automatically passed by the agent).
            title: The title for the Logseq page. This is also used for the filename.
            content_summary: The main textual content for the page. 
                             The LLM should embed Markdown links (e.g., [[Other Page Title]])
                             directly within this summary where appropriate for navigation.
            properties: A dictionary of properties to add to the page (e.g., {"type": "concept"}).
                        Defaults for 'type' ('concept') and 'source' ('chat_input') are applied.
            related_page_titles: A list of page titles that this page is conceptually related to.
                                 These will be formatted and added to a 'related::' property.

        Returns:
            A string confirming the action or reporting an error. This feedback is crucial for the LLM.
        """
        logfire.info(f"Tool 'create_logseq_page' invoked for title: '{title}'")
        
        filename = self._sanitize_filename(title)
        filepath = os.path.join(self.logseq_pages_path, f"{filename}.md")

        page_content_parts = []
        effective_properties = properties.copy() if properties else {}
        
        effective_properties.setdefault("type", "concept")
        effective_properties.setdefault("source", "chat_input")

        if related_page_titles:
            valid_related_titles = [rt.strip() for rt in related_page_titles if rt and rt.strip()]
            if valid_related_titles:
                related_links_str = ", ".join([f"[[{page_title}]]" for page_title in valid_related_titles])
                if "related" in effective_properties and effective_properties["related"]:
                    effective_properties["related"] = f"{effective_properties['related']}, {related_links_str}"
                else:
                    effective_properties["related"] = related_links_str
        
        for key, value in effective_properties.items():
            page_content_parts.append(f"{key}:: {value}")

        if page_content_parts:
            page_content_parts.append("") 

        page_content_parts.append(content_summary)
        final_page_content = "\n".join(page_content_parts)

        try:
            os.makedirs(self.logseq_pages_path, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(final_page_content)
            msg = f"Logseq page '{title}' (filename: '{filename}.md') created successfully in '{self.logseq_pages_path}'."
            logfire.info(msg)
            return msg
        except Exception as e:
            err_msg = f"Error creating Logseq page '{title}': {str(e)}"
            logfire.error(err_msg, exc_info=True)
            return f"Tool call failed for create_logseq_page (title: '{title}'). Error: {str(e)}"

    # Tool registration will be handled by _register_tools
    def search_logseq_graph(
        self,
        ctx: AnyRunContext, 
        query: str,
        search_in: Optional[List[SearchableLocation]] = None
    ) -> StructuredSearchResults:
        """
        Searches the Logseq graph for a given query string using Pydantic models.
        Can search in pages, journals, and identify TODO tasks.
        The search is case-insensitive.

        Args:
            ctx: The Pydantic AI run context (automatically passed by the agent).
            query: The text to search for.
            search_in: Optional list to specify search scope: ["pages", "journals", "tasks"].
                       If None or empty, searches pages and journals for general content, and all locations for tasks.
                       If "tasks" is specified, it will specifically look for task keywords.

        Returns:
            A StructuredSearchResults object containing found pages and journals.
        """
        logfire.info(f"Tool 'search_logseq_graph' invoked for query: '{query}', scope: {search_in}")
        
        page_results_map: Dict[str, PageMatch] = {}
        journal_results_map: Dict[str, JournalMatch] = {}
        # task_results: List[TaskMatch] = [] # Deferred for now

        lower_query = query.lower()
        task_keywords = ["TODO", "LATER", "NOW", "DOING", "DONE", "WAITING", "CANCELED"]
        lower_task_keywords = [kw.lower() for kw in task_keywords]

        effective_search_in = search_in if search_in else ["pages", "journals", "tasks"]

        files_to_process = []

        if "pages" in effective_search_in and os.path.exists(self.logseq_pages_path):
            for filename in os.listdir(self.logseq_pages_path):
                if filename.endswith(".md"):
                    files_to_process.append((os.path.join(self.logseq_pages_path, filename), "page", filename))
        
        if "journals" in effective_search_in and os.path.exists(self.logseq_journals_path):
            for filename in os.listdir(self.logseq_journals_path):
                if filename.endswith(".md"):
                    files_to_process.append((os.path.join(self.logseq_journals_path, filename), "journal", filename))

        for filepath, file_type, basename in files_to_process:
            page_title_from_filename = basename[:-3] # Remove .md
            try:
                # Filename (Title) matching for pages
                if file_type == "page" and lower_query in page_title_from_filename.lower():
                    if filepath not in page_results_map:
                        page_results_map[filepath] = PageMatch(
                            title=page_title_from_filename,
                            filename=basename,
                            path=filepath,
                            match_type="title"
                        )
                    else: # Already exists, ensure match_type reflects title match if it was content before
                        if page_results_map[filepath].match_type == "content":
                             page_results_map[filepath].match_type = "title" # Prioritize title match display

                # Content & Task matching
                with open(filepath, "r", encoding="utf-8") as f:
                    for i, line_content in enumerate(f):
                        line_lower = line_content.lower()
                        is_task_line = False
                        task_text_if_any = ""

                        # Task keyword check for this line
                        if "tasks" in effective_search_in:
                            for task_kw in lower_task_keywords:
                                if task_kw in line_lower.split(): # Check if keyword is a whole word
                                    is_task_line = True
                                    task_text_if_any = line_content.strip()
                                    break
                        
                        # General content query match OR if it's a task line containing the query
                        if lower_query in line_lower:
                            if file_type == "page":
                                if filepath not in page_results_map:
                                    page_results_map[filepath] = PageMatch(
                                        title=page_title_from_filename,
                                        filename=basename,
                                        path=filepath,
                                        # match_type will be title if already matched, else content
                                    )
                                if is_task_line:
                                    page_results_map[filepath].tasks_found.append(task_text_if_any)
                                else:
                                    page_results_map[filepath].content_snippets.append(line_content.strip())
                            
                            elif file_type == "journal":
                                date_str = basename[:-3] # e.g., 2024_11_25
                                if filepath not in journal_results_map:
                                    journal_results_map[filepath] = JournalMatch(
                                        date_str=date_str,
                                        filename=basename,
                                        path=filepath
                                    )
                                if is_task_line:
                                    journal_results_map[filepath].tasks_found.append(task_text_if_any)
                                else:
                                    journal_results_map[filepath].content_snippets.append(line_content.strip())                        
                        
                        # If it's a task line but general query wasn't in it, but we are searching for tasks broadly
                        # (e.g. query is "review" and line is "TODO fix bug", we might want to show all TODOs if search_in includes "tasks")
                        # This part is tricky: current logic finds tasks IF the query is ALSO in the line or if the query IS a task keyword.
                        # For now, task_found list is populated only if the line also matches the general query.

            except Exception as e:
                logfire.warning(f"Could not read or process file {filepath}: {e}")

        final_pages = [pm for pm in page_results_map.values() if pm.content_snippets or pm.tasks_found or pm.match_type == "title"]
        final_journals = list(journal_results_map.values())

        return StructuredSearchResults(
            query=query,
            pages=final_pages,
            journals=final_journals
        )

    # Tool registration will be handled by _register_tools
    def get_all_tasks(
        self,
        ctx: AnyRunContext,
        status_filter: Optional[List[str]] = None
    ) -> StructuredTaskResults:
        """
        Retrieves all tasks from the Logseq graph, optionally filtered by status.

        Args:
            ctx: The Pydantic AI run context.
            status_filter: A list of task statuses to filter by (e.g., ["TODO", "DONE"]). 
                           Case-insensitive. If None, all tasks are returned.

        Returns:
            A StructuredTaskResults object containing the found tasks.
        """
        logfire.info(f"Tool 'get_all_tasks' invoked. Status filter: {status_filter}")
        found_tasks: List[TaskMatch] = []
        task_keywords_map = {
            "TODO": "TODO", "LATER": "LATER", "NOW": "NOW", "DOING": "DOING", 
            "DONE": "DONE", "WAITING": "WAITING", "CANCELED": "CANCELED", "CANCELLED": "CANCELLED"
        } # Map various spellings/cases to a canonical status
        
        # Prepare status filter for case-insensitive comparison
        effective_status_filter: Optional[List[str]] = None
        if status_filter:
            effective_status_filter = [sf.upper() for sf in status_filter]

        files_to_scan = []
        if os.path.exists(self.logseq_pages_path):
            for filename in os.listdir(self.logseq_pages_path):
                if filename.endswith(".md"):
                    files_to_scan.append((os.path.join(self.logseq_pages_path, filename), "page", filename))
        
        if os.path.exists(self.logseq_journals_path):
            for filename in os.listdir(self.logseq_journals_path):
                if filename.endswith(".md"):
                    files_to_scan.append((os.path.join(self.logseq_journals_path, filename), "journal", filename))

        for filepath, file_type, basename in files_to_scan:
            page_title = basename[:-3] if file_type == "page" else None
            journal_date_str = basename[:-3] if file_type == "journal" else None
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for i, line_content in enumerate(f):
                        # Basic check for a task marker at the beginning of a line segment
                        # This regex looks for things like "- TODO", "* LATER", "TODO", etc.
                        # followed by some text.
                        match = re.match(r"^\s*[-*]?\s*([A-Z]+)\s+(.*)", line_content.strip())
                        if match:
                            task_status_candidate = match.group(1).upper()
                            task_description = match.group(2).strip()

                            canonical_status = task_keywords_map.get(task_status_candidate)

                            if canonical_status: # It's a recognized task status
                                if effective_status_filter is None or canonical_status in effective_status_filter:
                                    found_tasks.append(
                                        TaskMatch(
                                            description=task_description,
                                            status=canonical_status,
                                            source_filename=basename,
                                            source_path=filepath,
                                            source_type=file_type,
                                            line_number=i + 1,
                                            source_page_title=page_title,
                                            source_journal_date_str=journal_date_str
                                        )
                                    )
            except Exception as e:
                logfire.warning(f"Could not read or process file {filepath} for task extraction: {e}")

        return StructuredTaskResults(tasks=found_tasks, filter_applied=status_filter)

    def update_task_status(
        self,
        ctx: AnyRunContext,
        source_filename: str,
        line_number: int,
        new_status: str
    ) -> str:
        """
        Updates the status of a task in a Logseq file, supporting both keyword and checkbox formats.
        Args:
            ctx: The Pydantic AI run context.
            source_filename: The file where the task resides (e.g., 'Parlante Satelite.md').
            line_number: The 1-based line number of the task in the file.
            new_status: The new status (e.g., 'DONE', '[x]', 'TODO', '[ ]', etc.).
        Returns:
            A confirmation message or error.
        """
        logfire.info(f"Tool 'update_task_status' invoked for file: {source_filename}, line: {line_number}, new_status: {new_status}")
        abs_path = os.path.join(self.logseq_pages_path, source_filename)
        if not os.path.exists(abs_path):
            abs_path = os.path.join(self.logseq_journals_path, source_filename)
            if not os.path.exists(abs_path):
                return f"Error: File '{source_filename}' not found in pages or journals."
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            idx = line_number - 1
            if idx < 0 or idx >= len(lines):
                return f"Error: Line {line_number} is out of range in '{source_filename}'."
            original_line = lines[idx]
            # Regex for both keyword and checkbox tasks, preserving indentation
            task_pattern = re.compile(r"^(\s*[-*]?\s*)(TODO|DOING|LATER|NOW|WAITING|DONE|CANCELED|CANCELLED|\[ \]|\[x\]|\[/\])(\s+)(.*)$", re.IGNORECASE)
            match = task_pattern.match(original_line)
            if not match:
                return f"Error: Line {line_number} in '{source_filename}' does not appear to be a recognized Logseq task."
            indent, old_status, space, description = match.groups()
            # Normalize new_status: allow both keyword and checkbox
            valid_keywords = {"TODO", "DOING", "LATER", "NOW", "WAITING", "DONE", "CANCELED", "CANCELLED"}
            valid_checkboxes = {"[ ]", "[x]", "[/]"}
            ns = new_status.strip().upper()
            if ns in valid_keywords:
                new_status_str = ns
            elif ns in {v.upper() for v in valid_checkboxes}:
                # Preserve original case for checkboxes
                new_status_str = new_status.strip()
            else:
                return f"Error: '{new_status}' is not a recognized Logseq task status."
            new_line = f"{indent}{new_status_str}{space}{description}\n"
            lines[idx] = new_line
            with open(abs_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            return f"Task status updated: Line {line_number} in '{source_filename}' is now '{new_status_str}'."
        except Exception as e:
            logfire.error(f"Error updating task status in '{source_filename}', line {line_number}: {e}", exc_info=True)
            return f"Error updating task status: {str(e)}"

    def handle_search_query(self, query_text: str) -> str:
        """
        Processes a search query from the user by guiding the LLM to use the
        'search_logseq_graph' tool.

        Args:
            query_text: The user's search query.

        Returns:
            The search results or a confirmation/error message from the LLM.
        """
        logfire.info(f"Handling search query: '{query_text}'")

        system_prompt_text = (
            "You are TARS, a highly intelligent and articulate AI. Your responses are formal, concise, and task-oriented. "
            "Your current task is to use the 'search_logseq_graph' tool to find information "
            "within the Logseq graph based on the user's query. "
            "The tool will return a StructuredSearchResults object. You need to parse this object "
            "and present the findings in a clear, organized, and human-readable Markdown format. "
            "Group results by Pages and Journals. For each page, list its title (filename), then any content snippets and tasks found. "
            "For journals, list the date (filename), then content snippets and tasks."
        )

        user_prompt_text = f"""
        The user's input is: "{query_text}"
        Your task is to identify the actual search terms from the user's input. For example, if the user says "Busca Parlante Satelite", the core search terms are "Parlante Satelite".
        Then, use the 'search_logseq_graph' tool with these extracted search terms as the 'query' parameter.

        If relevant, you can also suggest a search scope (e.g., ["pages"], ["journals"], ["tasks"], or a combination) for the 'search_in' parameter of the tool, but this is optional.
        After the tool call (which returns a StructuredSearchResults object), format its `pages` and `journals` lists into a user-friendly Markdown response.
        If no results are found in the structured object, inform the user clearly.
        Example of desired output format if results are found:
        TARS > Affirmative. The search for "[Your Extracted Query]" yielded:

        **Pages:**
        *   **Page Title 1** (`filename1.md`)
            *   Content: "Snippet 1 from page 1..."
            *   Content: "Snippet 2 from page 1..."
            *   Task: "TODO: Task A on page 1"
        *   **Page Title 2** (`filename2.md`)
            *   Task: "DONE: Task B on page 2"

        **Journal Entries:**
        *   **Journal Date 1** (`date1.md`)
            *   Content: "Snippet from journal 1..."

        Please specify if further detail is required.
        """

        try:
            logfire.info("Sending search query to LLM to utilize 'search_logseq_graph' tool...")
            response = self.llm.run_sync(user_prompt_text, system_prompt=system_prompt_text)
            
            if response and hasattr(response, 'output') and response.output:
                final_llm_message = response.output
            elif response and hasattr(response, 'message') and response.message and hasattr(response.message, 'content'):
                final_llm_message = response.message.content
            else:
                logfire.warning(f"LLM response for search was empty or not in the expected format. Response: {response}")
                return "TARS > Search processing attempted. No clear response from LLM."

            logfire.info(f"LLM final response for search: '{final_llm_message}'")
            return final_llm_message.strip() if final_llm_message and final_llm_message.strip() else "TARS > Search executed. No specific textual response from LLM."

        except Exception as e:
            logfire.error(
                "Error during LLM processing for search query: {error_details}", 
                error_details=str(e), 
                exc_info=True
            )
            return f"TARS > System error during search processing: {str(e)}. Detailed diagnostics logged."

    def handle_get_all_tasks_query(self, user_query_text: Optional[str] = None) -> str:
        """
        Processes a user query to retrieve and list tasks, potentially filtered by status.
        Guides the LLM to use the 'get_all_tasks' tool and format its structured output.

        Args:
            user_query_text: The user's full query text, e.g., "show me all TODO tasks".

        Returns:
            A TARS-like formatted string listing tasks or a relevant message.
        """
        logfire.info(f"Handling 'get all tasks' query: '{user_query_text}'")

        system_prompt_text = (
            "You are TARS, a highly intelligent and articulate AI. Your responses are formal, concise, and task-oriented. "
            "Your current task is to use the 'get_all_tasks' tool to retrieve tasks from the Logseq graph. "
            "The user may specify a status to filter by (e.g., TODO, DONE). If so, pass this as the 'status_filter' list to the tool. "
            "If no specific status is mentioned, or if the user asks for 'all' tasks, call the tool without a 'status_filter' (or with None)."
            "The tool will return a StructuredTaskResults object containing a list of tasks. "
            "You need to parse this object and present the findings in a clear, organized, and human-readable Markdown format. "
            "Group tasks by their status. For each task, list its description, source (filename or journal date), and line number. "
            "If a filter was applied, mention it. If no tasks are found, state that clearly."
        )

        user_prompt_text = f"""
        The user's request is: "{user_query_text if user_query_text else 'list all tasks'}"
        
        1.  Determine if the user specified any task statuses to filter by (e.g., "TODO", "DONE", "pending", "completed"). 
            Recognize synonyms: "pending" might map to ["TODO", "LATER", "NOW", "DOING"]. "completed" to ["DONE"].
            If multiple statuses are implied, create a list for the 'status_filter' argument of the 'get_all_tasks' tool.
            If no status is specified or the user asks for 'all tasks', the 'status_filter' should be omitted (None).
        
        2.  Call the 'get_all_tasks' tool, providing the 'status_filter' if applicable.
        
        3.  The tool returns a `StructuredTaskResults` object. Format its `tasks` list into a user-friendly Markdown response.
            *   Clearly state if a filter was applied (using `filter_applied` from the results).
            *   Group tasks by their `status`.
            *   For each task, display its `description`, and its source (e.g., `[Page: page_title.md, line X]` or `[Journal: YYYY_MM_DD.md, line X]`).
            *   If no tasks are found for the given criteria, inform the user appropriately.

        Example of desired output format if tasks are found:
        TARS > Affirmative. Displaying tasks (Filter: [TODO, LATER]):

        **TODO:**
        *   Improve voice recognition [Parlante Satelite.md, line 7]
        *   Research Logseq API [research_notes.md, line 12]

        **LATER:**
        *   Refactor config module [dev_notes.md, line 23]

        Please specify if further actions are required.
        
        If no tasks are found:
        TARS > No tasks found matching your criteria (Filter: [COMPLETED]).
        """

        try:
            logfire.info(f"Sending 'get all tasks' request to LLM. User query: '{user_query_text}'")
            response = self.llm.run_sync(user_prompt_text, system_prompt=system_prompt_text)
            
            final_llm_message = ""
            if response and hasattr(response, 'output') and response.output:
                final_llm_message = response.output
            elif response and hasattr(response, 'message') and response.message and hasattr(response.message, 'content'):
                final_llm_message = response.message.content
            else:
                logfire.warning(f"LLM response for 'get all tasks' was empty/unexpected. Response: {response}")
                return "TARS > Task retrieval attempted. No clear response from LLM."

            logfire.info(f"LLM final response for 'get all tasks': '{final_llm_message}'")
            return final_llm_message.strip() if final_llm_message and final_llm_message.strip() else "TARS > Task list processed."

        except Exception as e:
            logfire.error(
                "Error during LLM processing for 'get all tasks' query: {error_details}", 
                error_details=str(e), 
                exc_info=True
            )
            return f"TARS > System error during task retrieval: {str(e)}. Detailed diagnostics logged."

    def process_text(self, text_block: str) -> str:
        """
        Processes a block of text using the LLM to analyze its content
        and create corresponding Logseq pages by calling the registered tools.

        Args:
            text_block: The input text to analyze and structure.

        Returns:
            A TARS-like confirmation message from the LLM after processing is complete.
        """
        logfire.info(f"Processing text block for Logseq structuring (first 100 chars): '{text_block[:100]}...'")

        system_prompt_text = (
            "You are TARS, a highly intelligent and articulate AI. Your responses are formal, concise, and task-oriented. "
            "You are currently tasked with analyzing provided text and structuring it into a Logseq knowledge base "
            "by intelligently using the 'create_logseq_page' tool. Adhere strictly to the tool's usage guidelines "
            "and the user's instructions for structuring the information. After all tool calls are completed, "
            "provide a brief, formal confirmation message about the overall action taken. "
            "If the user asks to search for something, use the 'search_logseq_graph' tool."
        )

        user_prompt_text = f"""
        Analyze the following text block. Your primary objective is to structure its content into a set of interconnected Logseq pages using the 'create_logseq_page' tool.

        Instructions for Page Creation:
        1.  **Identify Main Topic:** Determine the central concept or primary subject of the text.
        2.  **Extract Sub-Topics/Entities:** Identify key sub-topics, significant entities, or related concepts within the text that are distinct enough to warrant their own Logseq pages.
        3.  **Plan Page Creation:** For the main topic and each identified sub-topic/entity, meticulously plan the arguments for the 'create_logseq_page' tool:
            *   `title`: A concise and descriptive title for the Logseq page. This title will be used for `[[links]]`.
            *   `content_summary`: The main textual content for the page.
                *   For the **main topic's page**, this summary should provide an overview and explicitly include Markdown links (e.g., using bullet points like `- [[Name of Sub-Topic Page]]`) to the pages of its direct sub-topics or key related entities that you are also creating.
                *   For **sub-topic/entity pages**, the summary should contain specific information relevant to that sub-topic/entity. If it naturally relates back to the main topic or other sub-topics, include `[[links]]` to them within this summary.
            *   `properties`: A dictionary for page metadata. Ensure defaults like `{{\"type\": \"concept\", \"source\": \"chat_input\"}}` are effectively present. Add other relevant properties if clear from the text.
            *   `related_page_titles`: A list of strings, where each string is the exact title of another page that this page has a conceptual relationship with.
        4.  **Execute Tool Calls for Page Creation:** Make calls to `create_logseq_page` for the main topic and sub-topics.
        5.  **Interlinking:** Ensure pages are logically interlinked using `[[Page Title]]` syntax.

        The text to analyze for page creation is as follows:
        ---
        {text_block}
        ---
        After you have completed all necessary calls to 'create_logseq_page', provide a brief, formal confirmation message summarizing that the task has been executed. For example: "Page creation complete. Information structured into Logseq."
        Do NOT use the search_logseq_graph tool for this request. This request is about CREATING pages from the provided text.
        """

        try:
            logfire.info("Sending request to LLM for analysis and page creation...")
            response = self.llm.run_sync(user_prompt_text, system_prompt=system_prompt_text)
            
            if response and hasattr(response, 'output') and response.output:
                final_llm_message = response.output
            elif response and hasattr(response, 'message') and response.message and hasattr(response.message, 'content'):
                final_llm_message = response.message.content
            else:
                logfire.warning(f"LLM response was empty or not in the expected modern pydantic-ai format. Response: {response}")
                return "Processing attempted for page creation. Please check logs for details of tool execution."

            logfire.info(f"LLM final confirmation message for page creation: '{final_llm_message}'")
            return final_llm_message.strip() if final_llm_message and final_llm_message.strip() else "Page creation processing complete. Logseq structure generation initiated."

        except Exception as e:
            # Using template string for logfire.error
            logfire.error(
                "Error during LLM processing for page creation: {error_details}", 
                error_details=str(e), 
                exc_info=True
            )
            return f"TARS > System error during page creation: {str(e)}. Detailed diagnostics logged."

    def handle_update_task_status_query(self, user_query_text: str) -> str:
        """
        Orquesta la actualización de estado de una tarea Logseq a partir de una orden conversacional.
        1. Extrae del texto del usuario el nombre de la tarea y el nuevo estado (por defecto, si no se especifica, asume 'DONE').
        2. Busca la tarea usando get_all_tasks (sin filtro o con filtro flexible).
        3. Si hay coincidencia única, llama a update_task_status con los parámetros correctos.
        4. Si hay varias coincidencias, pide aclaración.
        5. Si no hay ninguna, informa.
        6. Responde siempre de forma formal y concisa (TARS).
        """
        logfire.info(f"Handling update task status query: '{user_query_text}'")

        system_prompt_text = (
            "Eres TARS, un asistente formal y conciso. Si el usuario pide marcar, completar, terminar o cambiar el estado de una tarea, "
            "debes: 1) extraer el nombre de la tarea y el estado deseado (por defecto, 'DONE' si no se especifica), "
            "2) buscar la tarea usando 'get_all_tasks' (sin filtro o con filtro flexible), "
            "3) si hay una coincidencia única, llamar a 'update_task_status' con el archivo, línea y estado correcto, "
            "4) si hay varias coincidencias, pedir aclaración al usuario mostrando las coincidencias, "
            "5) si no hay ninguna, informar formalmente. "
            "Siempre responde de forma formal, breve y precisa."
        )

        user_prompt_text = f"""
        El usuario ha dicho: "{user_query_text}"

        1. Extrae el nombre de la tarea y el estado deseado (por defecto, 'DONE' si no se especifica explícitamente otro estado como 'pendiente', 'en progreso', etc.).
        2. Busca la tarea usando 'get_all_tasks' (sin filtro o con filtro flexible si el usuario especifica el estado actual).
        3. Si hay una coincidencia única (por descripción, ignorando mayúsculas/minúsculas y espacios), llama a 'update_task_status' con:
            - source_filename
            - line_number
            - new_status (por defecto 'DONE' si no se especifica otro)
        4. Si hay varias coincidencias razonables, muestra una lista breve (descripción, archivo, línea) y pide al usuario que aclare cuál desea actualizar.
        5. Si no hay ninguna coincidencia, informa formalmente que no se encontró la tarea.
        6. Responde siempre como TARS, de forma formal y concisa.

        Ejemplo de respuesta si se actualiza:
        TARS > Tarea "Mejorar la voz" marcada como completada en [Parlante Satelite.md, línea 7]. ¿Desea realizar otra acción?

        Ejemplo si hay varias coincidencias:
        TARS > Se encontraron varias tareas que coinciden con "Mejorar la voz":
        * Mejorar la voz [Parlante Satelite.md, línea 7]
        * Mejorar la voz del asistente [Notas.md, línea 12]
        Por favor, indique con mayor precisión cuál desea actualizar.

        Ejemplo si no se encuentra:
        TARS > No se encontró ninguna tarea que coincida con "Mejorar la voz".
        """

        try:
            logfire.info("Enviando orden de actualización de tarea al LLM...")
            response = self.llm.run_sync(user_prompt_text, system_prompt=system_prompt_text)

            final_llm_message = ""
            if response and hasattr(response, 'output') and response.output:
                final_llm_message = response.output
            elif response and hasattr(response, 'message') and response.message and hasattr(response.message, 'content'):
                final_llm_message = response.message.content
            else:
                logfire.warning(f"LLM response for update task status was empty/unexpected. Response: {response}")
                return "TARS > Actualización de tarea intentada. No se obtuvo respuesta clara del LLM."

            logfire.info(f"LLM final response for update task status: '{final_llm_message}'")
            return final_llm_message.strip() if final_llm_message and final_llm_message.strip() else "TARS > Actualización de tarea procesada."

        except Exception as e:
            logfire.error(
                "Error durante el procesamiento LLM para actualización de tarea: {error_details}", 
                error_details=str(e), 
                exc_info=True
            )
            return f"TARS > Error del sistema durante la actualización de tarea: {str(e)}. Diagnóstico detallado en logs." 

    def get_task_logbook(
        self,
        ctx: AnyRunContext,
        source_filename: str,
        line_number: int
    ) -> TaskLogbookResult:
        """
        Extrae el contenido de :LOGBOOK: para una tarea específica en un archivo Logseq.
        Args:
            ctx: Contexto de ejecución Pydantic AI.
            source_filename: Nombre del archivo donde está la tarea.
            line_number: Línea (1-based) donde está la tarea.
        Returns:
            TaskLogbookResult con el contenido de :LOGBOOK: (clock_lines y raw_logbook).
        """
        abs_path = os.path.join(self.logseq_pages_path, source_filename)
        if not os.path.exists(abs_path):
            abs_path = os.path.join(self.logseq_journals_path, source_filename)
            if not os.path.exists(abs_path):
                return TaskLogbookResult(
                    description="",
                    status="",
                    source_filename=source_filename,
                    line_number=line_number,
                    logbook=TaskLogbookEntry(),
                    error=f"Archivo '{source_filename}' no encontrado."
                )
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            idx = line_number - 1
            if idx < 0 or idx >= len(lines):
                return TaskLogbookResult(
                    description="",
                    status="",
                    source_filename=source_filename,
                    line_number=line_number,
                    logbook=TaskLogbookEntry(),
                    error=f"Línea {line_number} fuera de rango en '{source_filename}'."
                )
            # Buscar el bloque :LOGBOOK: debajo de la tarea
            logbook_lines = []
            in_logbook = False
            for l in lines[idx+1:]:
                if l.strip().startswith(":LOGBOOK:"):
                    in_logbook = True
                    continue
                if in_logbook:
                    if l.strip().startswith(":END:"):
                        break
                    logbook_lines.append(l.rstrip())
                # Si hay otro bloque o tarea, salir
                if not in_logbook and (l.strip().startswith("- ") or l.strip().startswith("* ")):
                    break
            # Extraer descripción y status de la tarea
            match = re.match(r"^\s*[-*]?\s*([A-Z]+)\s+(.*)", lines[idx].strip())
            status = match.group(1) if match else ""
            description = match.group(2) if match else lines[idx].strip()
            return TaskLogbookResult(
                description=description,
                status=status,
                source_filename=source_filename,
                line_number=line_number,
                logbook=TaskLogbookEntry(clock_lines=logbook_lines, raw_logbook="\n".join(logbook_lines)),
                error=None
            )
        except Exception as e:
            return TaskLogbookResult(
                description="",
                status="",
                source_filename=source_filename,
                line_number=line_number,
                logbook=TaskLogbookEntry(),
                error=str(e)
            )

    def search_tasks(
        self,
        ctx: AnyRunContext,
        query: str,
        status_filter: Optional[list[str]] = None,
        tag_filter: Optional[str] = None
    ) -> SearchTasksResult:
        """
        Busca tareas en el grafo Logseq por texto y/o etiqueta, con opción de filtrar por estado.
        Args:
            ctx: Contexto de ejecución Pydantic AI.
            query: Texto a buscar en la descripción de la tarea.
            status_filter: Lista de estados de tarea a filtrar (opcional).
            tag_filter: Etiqueta (sin #) a filtrar (opcional).
        Returns:
            SearchTasksResult con la lista de tareas encontradas.
        """
        found_tasks: list[TaskMatch] = []
        task_keywords_map = {
            "TODO": "TODO", "LATER": "LATER", "NOW": "NOW", "DOING": "DOING",
            "DONE": "DONE", "WAITING": "WAITING", "CANCELED": "CANCELED", "CANCELLED": "CANCELLED"
        }
        effective_status_filter = [sf.upper() for sf in status_filter] if status_filter else None
        files_to_scan = []
        if os.path.exists(self.logseq_pages_path):
            for filename in os.listdir(self.logseq_pages_path):
                if filename.endswith(".md"):
                    files_to_scan.append((os.path.join(self.logseq_pages_path, filename), "page", filename))
        if os.path.exists(self.logseq_journals_path):
            for filename in os.listdir(self.logseq_journals_path):
                if filename.endswith(".md"):
                    files_to_scan.append((os.path.join(self.logseq_journals_path, filename), "journal", filename))
        for filepath, file_type, basename in files_to_scan:
            page_title = basename[:-3] if file_type == "page" else None
            journal_date_str = basename[:-3] if file_type == "journal" else None
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for i, line_content in enumerate(f):
                        match = re.match(r"^\s*[-*]?\s*([A-Z]+)\s+(.*)", line_content.strip())
                        if match:
                            task_status_candidate = match.group(1).upper()
                            task_description = match.group(2).strip()
                            canonical_status = task_keywords_map.get(task_status_candidate)
                            if canonical_status:
                                if effective_status_filter and canonical_status not in effective_status_filter:
                                    continue
                                if query and query.lower() not in task_description.lower():
                                    continue
                                if tag_filter and f"#{tag_filter.lower()}" not in task_description.lower():
                                    continue
                                found_tasks.append(
                                    TaskMatch(
                                        description=task_description,
                                        status=canonical_status,
                                        source_filename=basename,
                                        source_path=filepath,
                                        source_type=file_type,
                                        line_number=i + 1,
                                        source_page_title=page_title,
                                        source_journal_date_str=journal_date_str
                                    )
                                )
            except Exception as e:
                continue
        return SearchTasksResult(tasks=found_tasks, query=query, status_filter=status_filter, tag_filter=tag_filter) 

    def handle_task_logbook_query(self, user_query_text: str) -> str:
        """
        Orquesta la consulta de historial/logbook de una tarea a partir de una orden conversacional.
        1. Extrae del texto del usuario el nombre de la tarea (o referencia).
        2. Busca la tarea usando search_tasks (sin filtro o con filtro flexible).
        3. Si hay coincidencia única, llama a get_task_logbook con los parámetros correctos.
        4. Si hay varias coincidencias, muestra la lista breve (descripción, archivo, línea) y espera aclaración del usuario.
        5. Si el usuario responde con un número, archivo+línea o referencia, muestra el logbook correspondiente.
        6. Si no hay ninguna coincidencia, informa formalmente.
        7. Responde siempre de forma formal y concisa (TARS).
        """
        logfire.info(f"Handling task logbook query: '{user_query_text}'")

        system_prompt_text = (
            "Eres TARS, un asistente formal y conciso. Si el usuario pide el historial, logbook o seguimiento de una tarea, "
            "debes: 1) extraer el nombre o referencia de la tarea, 2) buscar la tarea usando 'search_tasks' (sin filtro salvo que el usuario especifique), "
            "3) si hay una coincidencia única, llama a 'get_task_logbook' con el archivo y línea correctos, "
            "4) si hay varias coincidencias, muestra una lista breve (descripción, archivo, línea) y espera que el usuario aclare a cuál se refiere (por número, archivo+línea, etc.), "
            "5) si el usuario responde con una referencia válida, muestra el logbook correspondiente, "
            "6) si no hay ninguna coincidencia, informa formalmente. "
            "Siempre responde de forma formal, breve y precisa. "
            "Mantén el contexto conversacional para que el usuario pueda referirse a la tarea por número, descripción parcial, archivo o línea en turnos siguientes."
        )

        user_prompt_text = f"""
        El usuario ha dicho: "{user_query_text}"

        1. Si el usuario pide el historial, logbook o seguimiento de una tarea, busca la tarea por nombre o referencia usando 'search_tasks'.
        2. Si hay una sola coincidencia, llama a 'get_task_logbook' con el archivo y línea correctos y muestra el resultado.
        3. Si hay varias coincidencias, muestra una lista numerada (descripción, archivo, línea) y espera que el usuario aclare a cuál se refiere (por número, archivo+línea, etc.).
        4. Si el usuario responde con una referencia válida, muestra el logbook correspondiente.
        5. Si no hay ninguna coincidencia, informa formalmente.
        6. Siempre responde como TARS, de forma formal y concisa.

        Ejemplo de respuesta si se muestra el logbook:
        TARS > LOGBOOK de la tarea "Mejorar la voz" [TODO] en Parlante Satelite.md, línea 8:
          CLOCK: [2025-05-11 Sun 10:58:14]--[2025-05-11 Sun 10:58:16] =>  00:00:02
          ...

        Ejemplo si hay varias coincidencias:
        TARS > Se encontraron varias tareas que coinciden con "Mejorar la voz":
        1. Mejorar la voz [Parlante Satelite.md, línea 8]
        2. Mejorar la voz del asistente [Notas.md, línea 12]
        Por favor, indique el número o referencia de la tarea cuyo historial desea ver.

        Ejemplo si no se encuentra:
        TARS > No se encontró ninguna tarea que coincida con "Mejorar la voz".
        """

        try:
            logfire.info("Enviando consulta de logbook de tarea al LLM...")
            response = self.llm.run_sync(user_prompt_text, system_prompt=system_prompt_text)

            final_llm_message = ""
            if response and hasattr(response, 'output') and response.output:
                final_llm_message = response.output
            elif response and hasattr(response, 'message') and response.message and hasattr(response.message, 'content'):
                final_llm_message = response.message.content
            else:
                logfire.warning(f"LLM response for task logbook query was empty/unexpected. Response: {response}")
                return "TARS > Consulta de historial de tarea intentada. No se obtuvo respuesta clara del LLM."

            logfire.info(f"LLM final response for task logbook query: '{final_llm_message}'")
            return final_llm_message.strip() if final_llm_message and final_llm_message.strip() else "TARS > Consulta de historial de tarea procesada."

        except Exception as e:
            logfire.error(
                "Error durante el procesamiento LLM para consulta de logbook de tarea: {error_details}", 
                error_details=str(e), 
                exc_info=True
            )
            return f"TARS > Error del sistema durante la consulta de historial de tarea: {str(e)}. Diagnóstico detallado en logs." 