from utils.llm_config import get_llm_client
from langchain_core.messages import HumanMessage
import json
import re


class SparkSQLGenerator:
    """
    Generate Spark SQL (HiveQL-compatible) transformation code using Azure OpenAI.
    Produces CREATE TABLE AS SELECT and VIEW definitions suitable for Spark SQL / Databricks notebooks.
    """

    def __init__(self):
        self.llm_client = get_llm_client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_sql(
        self,
        input_columns: list[str],
        target_columns: list[str],
        transformation_logic: dict[str, str],
        source_table: str = "source_data",
        target_table: str = "transformed_data",
        crosswalk_tables: list[dict] | None = None,
        sample_records: list[dict] | None = None,
        progress_callback=None,
    ) -> dict:
        """
        Generate Spark SQL for the given transformation spec.

        Returns dict with keys:
            - 'ddl_sql': str       — CREATE TABLE AS SELECT statement
            - 'view_sql': str      — CREATE OR REPLACE VIEW statement
            - 'select_sql': str    — Plain SELECT query (runnable in any SQL context)
            - 'column_exprs': dict — target_col -> SQL expression
            - 'error': str | None
        """
        print("DEBUG: Starting Spark SQL generation")

        if progress_callback:
            progress_callback(0, 2, "Building SQL prompt…")

        prompt = self._build_prompt(
            input_columns, target_columns, transformation_logic,
            source_table, target_table, crosswalk_tables, sample_records
        )

        if progress_callback:
            progress_callback(1, 2, "Generating Spark SQL with AI…")

        try:
            response = self.llm_client.invoke([HumanMessage(content=prompt)])
            response_text = (
                str(response.content) if hasattr(response, "content") else str(response)
            )
            result = self._parse_response(response_text)

            if progress_callback:
                progress_callback(2, 2, "Spark SQL ready!")

            return result

        except Exception as e:
            print(f"DEBUG: SQL generation failed: {e}")
            return {
                "ddl_sql": f"-- Generation failed: {e}",
                "view_sql": f"-- Generation failed: {e}",
                "select_sql": f"-- Generation failed: {e}",
                "column_exprs": {},
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        input_columns: list[str],
        target_columns: list[str],
        transformation_logic: dict[str, str],
        source_table: str,
        target_table: str,
        crosswalk_tables: list[dict] | None,
        sample_records: list[dict] | None,
    ) -> str:
        prompt = f"""You are an expert in Spark SQL / HiveQL. Generate clean, production-ready Spark SQL for the transformation described below.

=== SOURCE TABLE: `{source_table}` ===
Input columns: {json.dumps(input_columns)}

=== TARGET TABLE: `{target_table}` ===
Output columns: {json.dumps(target_columns)}

=== TRANSFORMATION LOGIC (English) ===
"""
        for col in target_columns:
            logic = transformation_logic.get(col, "Pass through or leave empty.")
            prompt += f"  {col}: {logic}\n"

        if crosswalk_tables:
            prompt += "\n=== CROSSWALK / LOOKUP TABLES ===\n"
            for cw in crosswalk_tables:
                prompt += f"\nTable: `{cw['name']}`  |  Columns: {json.dumps(cw['columns'])}\n"
                prompt += f"Sample rows: {json.dumps(cw['sample_data'][:3])}\n"
            prompt += (
                "\nINSTRUCTIONS: Use LEFT JOIN to these tables for lookups. "
                "Handle non-matching rows with COALESCE or CASE WHEN.\n"
            )

        if sample_records:
            prompt += f"\n=== SAMPLE INPUT RECORDS ===\n{json.dumps(sample_records[:3], indent=2)}\n"

        prompt += f"""
=== REQUIRED RESPONSE FORMAT ===
Respond with ONLY this JSON (no prose outside the JSON block):

```json
{{
  "column_exprs": {{
    "<target_col>": "<SQL expression for this column, e.g. UPPER(first_name) AS full_name>"
  }},
  "select_sql": "<Plain SELECT statement with all column transformations, FROM {source_table}. Add any JOIN clauses needed.>",
  "view_sql": "CREATE OR REPLACE VIEW {target_table}_vw AS\\n<full SELECT statement>",
  "ddl_sql": "CREATE TABLE IF NOT EXISTS {target_table}\\nUSING DELTA\\nAS\\n<full SELECT statement>"
}}
```

=== CODING STANDARDS ===
1. Use standard Spark SQL / HiveQL functions: UPPER, LOWER, TRIM, CONCAT, COALESCE, CAST, DATE_FORMAT, DATEDIFF, CASE WHEN, REGEXP_REPLACE, SPLIT, etc.
2. Always alias computed expressions: `expr AS target_col_name`.
3. Handle NULLs with COALESCE or CASE WHEN IS NULL THEN ... END.
4. For crosswalk lookups, use LEFT JOIN on the lookup table name.
5. `ddl_sql` should use Delta format (`USING DELTA`) for Databricks compatibility.
6. Add SQL comments (`-- `) above each column expression explaining the logic.
7. Return ONLY the JSON block.
"""
        return prompt

    # ------------------------------------------------------------------
    # Response parser
    # ------------------------------------------------------------------

    def _parse_response(self, response_text: str) -> dict:
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        raw_json = json_match.group(1) if json_match else response_text.strip()
        raw_json = re.sub(r"^```[a-zA-Z]*\n?", "", raw_json)
        raw_json = re.sub(r"\n?```$", "", raw_json)

        try:
            data = json.loads(raw_json)
            return {
                "column_exprs": data.get("column_exprs", {}),
                "select_sql": data.get("select_sql", ""),
                "view_sql": data.get("view_sql", ""),
                "ddl_sql": data.get("ddl_sql", ""),
                "error": None,
            }
        except json.JSONDecodeError:
            # Try extracting raw SQL block
            sql_match = re.search(r"```sql\s*(.*?)```", response_text, re.DOTALL)
            sql = sql_match.group(1).strip() if sql_match else response_text.strip()
            return {
                "column_exprs": {},
                "select_sql": sql,
                "view_sql": sql,
                "ddl_sql": sql,
                "error": "Could not parse structured JSON — raw SQL extracted.",
            }
