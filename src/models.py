from typing import List, Optional, Dict
from pydantic import BaseModel

class Task(BaseModel):
    """
    Modelo base para representar una tarea genérica en Logseq.
    """
    description: str
    status: str
    tags: Optional[List[str]] = None

class PagePlan(BaseModel):
    title: str
    content_summary: str
    properties: Optional[Dict[str, str]] = None
    related_page_titles: Optional[List[str]] = None

class KnowledgeGraphPlan(BaseModel):
    main_topic: str
    pages: List[PagePlan] 