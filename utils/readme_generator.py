"""
readme_generator.py
--------------------
Generates a professional technical README.md for the data pipeline
that was just built — including schema, transformation logic,
how to run, and dependencies.
"""

from utils.llm_config import get_llm_client
from langchain_core.messages import HumanMessage
import json


class ReadmeGenerator:
    """
    AI-powered pipeline documentation generator.
    Produces a markdown README suitable for a GitHub repo.
    """

    def __init__(self):
        self.llm_client = get_llm_client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        input_columns: list[str],
        target_columns: list[str],
        transformation_logic: dict[str, str],
        crosswalk_tables: list[dict] | None = None,
        quality_rules: list[dict] | None = None,
        pipeline_name: str = "DataForge Pipeline",
    ) -> str:
        """
        Generate a markdown README for this pipeline.

        Returns:
            str — full markdown content ready to save as README.md
        """
        prompt = self._build_prompt(
            input_columns, target_columns, transformation_logic,
            crosswalk_tables, quality_rules, pipeline_name
        )

        try:
            response = self.llm_client.invoke([HumanMessage(content=prompt)])
            raw = str(response.content) if hasattr(response, "content") else str(response)
            # Strip any triple-backtick markdown wrapper if the model added one
            raw = raw.strip()
            if raw.startswith("```markdown"):
                raw = raw[len("```markdown"):].strip()
            if raw.startswith("```"):
                raw = raw[3:].strip()
            if raw.endswith("```"):
                raw = raw[:-3].strip()
            return raw
        except Exception as e:
            return f"# Pipeline README\n\n> README generation failed: {e}\n"

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        input_columns: list[str],
        target_columns: list[str],
        transformation_logic: dict[str, str],
        crosswalk_tables: list[dict] | None,
        quality_rules: list[dict] | None,
        pipeline_name: str,
    ) -> str:

        logic_lines = "\n".join(
            f"- **{col}**: {logic}"
            for col, logic in transformation_logic.items()
            if logic.strip()
        )

        cw_section = ""
        if crosswalk_tables:
            cw_section = "\n### Lookup / Crosswalk Tables\n"
            for cw in crosswalk_tables:
                cw_section += f"- **{cw['name']}**: columns `{', '.join(cw.get('columns', []))}`\n"

        dq_section = ""
        if quality_rules:
            dq_section = "\n### Data Quality Rules\n| Column | Rule | Description |\n|--------|------|-------------|\n"
            for r in quality_rules:
                dq_section += f"| `{r.get('column','—')}` | {r.get('rule','—')} | {r.get('description','—')} |\n"

        prompt = f"""You are a Senior Data Engineer writing documentation. Write a professional, thorough README.md for the following data pipeline.

=== PIPELINE INFO ===
Name: {pipeline_name}
Input columns:  {json.dumps(input_columns)}
Target columns: {json.dumps(target_columns)}

=== TRANSFORMATION LOGIC ===
{logic_lines}
{cw_section}
{dq_section}

=== INSTRUCTIONS ===
Write a complete README.md in Markdown. Include ALL of these sections:

1. **Title & badges** — pipeline name, Python badge, Spark badge, Streamlit badge, license badge
2. **Overview** — 2–3 sentence description of what this pipeline does
3. **Pipeline Architecture** — a simple ASCII diagram showing: Source → Quality Check → Transform → Output
4. **Schema**
   - Source schema table (column name, type)
   - Target schema table (column name, type, derivation)
5. **Transformation Logic** — detailed table showing each target column and how it's derived
6. **Data Quality Rules** — if provided, list all rules in a table
7. **Lookup Tables** — if crosswalks provided, describe each
8. **How to Run**
   - Prerequisites (Python 3.11+, Spark, Groq API key)
   - Installation steps
   - Running locally with Streamlit
   - Running on Spark cluster
9. **Environment Variables** — table of required env vars with descriptions
10. **Project Structure** — file tree
11. **Tech Stack** — table of technologies used
12. **License** — MIT

Write in a professional, confident tone. Use tables, code blocks, and emoji headers.
Return ONLY the markdown — no extra commentary.
"""
        return prompt
