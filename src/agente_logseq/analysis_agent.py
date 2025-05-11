from typing import Optional, List
from src.models import KnowledgeGraphPlan, PagePlan
import json
import openai
from pydantic import ValidationError
from src.agente_logseq.config import settings
import logfire
import re

class AnalysisAgent:
    """
    Agente encargado de analizar texto y estructurarlo en un KnowledgeGraphPlan para Logseq.
    """
    def __init__(self):
        openai.api_key = settings.OPENAI_API_KEY
        self.model = "gpt-4o-mini"

    def analyze_text(self, text: str, main_topic: Optional[str] = None) -> KnowledgeGraphPlan:
        """
        Analiza el texto y devuelve un KnowledgeGraphPlan.
        Por ahora, es un stub que genera un ejemplo fijo.
        """
        # TODO: Integrar LLM o lógica real de análisis
        if not main_topic:
            main_topic = "Tema Principal"
        pages = [
            PagePlan(
                title=main_topic,
                content_summary="Resumen generado automáticamente.",
                properties={"type": "concepto", "source": "analysis_agent"},
                related_page_titles=[]
            )
        ]
        return KnowledgeGraphPlan(main_topic=main_topic, pages=pages)

    def analyze_text_to_graph_plan(self, text: str) -> KnowledgeGraphPlan:
        prompt = (
            "Eres un asistente experto en estructuración de conocimiento para Logseq.\n"
            "Analiza el siguiente texto y devuélvelo como un JSON válido siguiendo este modelo Pydantic:\n"
            "KnowledgeGraphPlan(main_topic: str, pages: List[PagePlan(title: str, content_summary: str, properties: dict, related_page_titles: List[str])])\n"
            "IMPORTANTE: Siempre debes crear una página principal (padre) cuyo título sea el tema principal del texto.\n"
            "Todas las demás páginas deben ser secundarias (subtemas) y estar vinculadas a la página principal usando el campo 'related_page_titles'.\n"
            "La página principal debe contener enlaces a todas las secundarias, y las secundarias deben estar relacionadas con la principal.\n"
            "En el campo 'properties' (dict), TODOS los valores deben ser strings.\n"
            "Si una propiedad tiene varios valores (por ejemplo, ventajas), concaténalos en un solo string, separados por punto y coma.\n"
            "SOLO debes estructurar y jerarquizar exactamente la información y los títulos que aparecen en el texto proporcionado.\n"
            "NO inventes subtemas, páginas ni contenido adicional. NO agregues subpáginas, ejemplos, ni subdividas temas a menos que estén explícitamente en el texto.\n"
            "NO completes ni expandas la información más allá de lo dado.\n"
            "Ejemplo: 'ventajas': 'Mayor precisión; Actualidad; Relevancia contextual'\n"
            "Texto a analizar:\n"
            f"{text}\n"
            "Devuelve SOLO el JSON, sin explicaciones."
        )
        logfire.info(f"Prompt enviado al LLM:\n{prompt}")
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1500,
            )
            json_str = response.choices[0].message.content.strip()
            logfire.info(f"Respuesta cruda del LLM:\n{json_str}")
            # Limpieza de bloque de código Markdown
            if json_str.startswith("```"):
                json_str = re.sub(r"^```[a-zA-Z]*\n?", "", json_str)
                json_str = re.sub(r"```$", "", json_str)
                json_str = json_str.strip()
            plan = KnowledgeGraphPlan.model_validate_json(json_str)
            logfire.info(f"KnowledgeGraphPlan validado correctamente.")
        except ValidationError as e:
            logfire.error(f"Error validando el JSON generado por el LLM: {e}")
            raise ValueError(f"Error validando el JSON generado por el LLM: {e}")
        except Exception as e:
            logfire.error(f"Error al llamar al LLM: {e}")
            raise
        return plan

    def save_plan_to_file(self, plan: KnowledgeGraphPlan, path: str):
        """
        Guarda el KnowledgeGraphPlan como JSON en el archivo especificado.
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(plan.json(indent=2, ensure_ascii=False))

def is_analyze_text_query(text: str) -> bool:
    text_lower = text.lower()
    return text_lower.startswith("analizar texto ")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4 and sys.argv[1] == "analizar" and sys.argv[2] == "texto":
        input_path = sys.argv[3]
        output_path = sys.argv[4] if len(sys.argv) > 4 else "plan.json"
        agent = AnalysisAgent()
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        plan = agent.analyze_text_to_graph_plan(text)
        agent.save_plan_to_file(plan, output_path)
        print(f"Plan guardado en {output_path}")
    else:
        print("Uso: python analysis_agent.py analizar texto <ruta_entrada> <ruta_salida>") 