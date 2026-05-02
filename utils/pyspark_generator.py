from utils.llm_config import get_llm_client
from langchain_core.messages import HumanMessage
import json
import re


class PySparkGenerator:
    """
    Generate production-ready PySpark (Python) transformation code using Azure OpenAI.
    Mirrors ScalaCodeGenerator but targets the Python/PySpark API.
    """

    def __init__(self, max_iterations: int = 3):
        self.llm_client = get_llm_client()
        self.max_iterations = max_iterations

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_pyspark_code(
        self,
        input_columns: list[str],
        target_columns: list[str],
        transformation_logic: dict[str, str],
        sample_records: list[dict] | None = None,
        crosswalk_tables: list[dict] | None = None,
        progress_callback=None,
    ) -> dict:
        """
        Generate PySpark code for the given transformation spec.

        Returns dict with keys:
            - 'full_code': str  — complete, runnable PySpark module
            - 'per_column_code': dict — column -> withColumn expression
            - 'sample_output': list[dict]
            - 'iterations_used': int
            - 'error': str | None
        """
        print("DEBUG: Starting PySpark generation")

        if progress_callback:
            progress_callback(0, self.max_iterations + 1, "Building PySpark prompt…")

        prompt = self._build_prompt(
            input_columns, target_columns, transformation_logic,
            sample_records, crosswalk_tables
        )

        iteration = 0
        response_text = ""
        last_error = None

        while iteration < self.max_iterations:
            iteration += 1
            print(f"DEBUG: PySpark iteration {iteration}")

            if progress_callback:
                progress_callback(
                    iteration, self.max_iterations + 1,
                    f"Generating PySpark code (attempt {iteration})…"
                )

            try:
                current_prompt = (
                    prompt if iteration == 1
                    else self._refinement_prompt(prompt, response_text, last_error)
                )
                response = self.llm_client.invoke([HumanMessage(content=current_prompt)])
                response_text = (
                    str(response.content) if hasattr(response, "content") else str(response)
                )
                result = self._parse_response(response_text, target_columns)

                if progress_callback:
                    progress_callback(
                        self.max_iterations + 1, self.max_iterations + 1,
                        "PySpark code ready!"
                    )

                result["iterations_used"] = iteration
                return result

            except Exception as e:
                last_error = str(e)
                print(f"DEBUG: PySpark iteration {iteration} failed: {last_error}")
                if iteration >= self.max_iterations:
                    return {
                        "full_code": response_text or f"# Generation failed: {last_error}",
                        "per_column_code": {},
                        "sample_output": [],
                        "iterations_used": iteration,
                        "error": last_error,
                    }

        return {
            "full_code": "# No code generated",
            "per_column_code": {},
            "sample_output": [],
            "iterations_used": iteration,
        }

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        input_columns: list[str],
        target_columns: list[str],
        transformation_logic: dict[str, str],
        sample_records: list[dict] | None,
        crosswalk_tables: list[dict] | None,
    ) -> str:
        prompt = f"""You are an expert PySpark Python developer. Generate clean, production-ready PySpark code that transforms a source DataFrame into a target DataFrame.

=== INPUT COLUMNS (Source DataFrame) ===
{json.dumps(input_columns, indent=2)}

=== TARGET COLUMNS (Output DataFrame) ===
{json.dumps(target_columns, indent=2)}

=== TRANSFORMATION LOGIC (English descriptions) ===
"""
        for col in target_columns:
            logic = transformation_logic.get(col, "Pass through or leave empty.")
            prompt += f"  {col}: {logic}\n"

        if crosswalk_tables:
            prompt += "\n=== CROSSWALK / LOOKUP TABLES ===\n"
            for cw in crosswalk_tables:
                prompt += f"\nTable: {cw['name']}  |  Columns: {json.dumps(cw['columns'])}\n"
                prompt += f"Sample rows: {json.dumps(cw['sample_data'][:3], indent=2)}\n"
            prompt += (
                "\nINSTRUCTIONS FOR CROSSWALK TABLES:\n"
                "- Accept each crosswalk table as a DataFrame parameter to the transform function.\n"
                "- Use F.broadcast() for small lookup tables.\n"
                "- Handle non-matching rows with left join + coalesce/fillna.\n"
            )

        if sample_records:
            prompt += f"\n=== SAMPLE INPUT RECORDS ===\n{json.dumps(sample_records[:5], indent=2)}\n"

        prompt += """
=== REQUIRED RESPONSE FORMAT ===
Respond with ONLY this JSON structure (no prose, no markdown outside the JSON):

```json
{
  "per_column_code": {
    "<target_col>": "<single df.withColumn(...) expression or F.col(...) expression>"
  },
  "full_code": "<Complete PySpark Python module as a single string. Must include: imports, type hints, docstring, transform() function, and __main__ block with SparkSession example.>",
  "sample_output": [
    {"<target_col_1>": "<computed_value>", ...}
  ]
}
```

=== CODING STANDARDS ===
1. Use `from pyspark.sql import functions as F, DataFrame, SparkSession, Window`.
2. Add type hints: `def transform(df: DataFrame, ...) -> DataFrame`.
3. Chain `.withColumn()` calls cleanly; use `F.col`, `F.lit`, `F.when`, `F.coalesce`, `F.concat`, `F.split`, `F.regexp_replace`, `F.to_date`, `F.date_format`, `F.datediff`, etc.
4. Null safety: always `.otherwise(F.lit(None))` or `.otherwise(F.lit(""))` at end of `when` chains.
5. Include module-level docstring describing the transformation.
6. Include a `__main__` block that builds a SparkSession and shows usage with a tiny synthetic DataFrame.
7. For crosswalk lookups, include them as named parameters: `def transform(df, gender_lookup=None, ...)`.
8. Select only target columns at the end: `return df.select([F.col(c) for c in TARGET_COLUMNS])`.
9. `sample_output`: apply transformations mentally to the sample_records; return [] if no sample_records.
10. Return ONLY the JSON block.
"""
        return prompt

    def _refinement_prompt(self, original: str, prev_response: str, error: str) -> str:
        return (
            f"{original}\n\n"
            f"=== PREVIOUS ATTEMPT (FAILED) ===\n{prev_response[:2000]}\n\n"
            f"=== ERROR ===\n{error}\n\n"
            "Please correct the issues and return a valid JSON response."
        )

    # ------------------------------------------------------------------
    # Response parser
    # ------------------------------------------------------------------

    def _parse_response(self, response_text: str, target_columns: list[str]) -> dict:
        # Try to extract JSON block
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        raw_json = json_match.group(1) if json_match else response_text.strip()

        # Strip stray markdown fences
        raw_json = re.sub(r"^```[a-zA-Z]*\n?", "", raw_json)
        raw_json = re.sub(r"\n?```$", "", raw_json)

        # Remove invalid control characters
        raw_json = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw_json)

        try:
            data = json.loads(raw_json)
            full_code = data.get("full_code", response_text)
            if isinstance(full_code, str):
                full_code = full_code.replace("\\n", "\n").replace("\\t", "\t")
            return {
                "full_code": full_code,
                "per_column_code": data.get("per_column_code", {}),
                "sample_output": data.get("sample_output", []),
            }
        except json.JSONDecodeError:
            # Try to extract Python code block as fallback
            code_match = re.search(r"```python\s*(.*?)```", response_text, re.DOTALL)
            code = code_match.group(1).strip() if code_match else response_text.strip()
            return {
                "full_code": code,
                "per_column_code": {},
                "sample_output": [],
            }
