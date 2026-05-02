from utils.llm_config import get_llm_client
from langchain_core.messages import HumanMessage
import json
import re


class ScalaCodeGenerator:
    """
    Generate Scala transformation code using Azure OpenAI LLM.
    Produces a Scala function/code for each target column based on
    input columns and transformation logic described in plain English.
    Supports crosswalk/lookup tables for value mapping.
    """

    def __init__(self, max_iterations=3):
        self.llm_client = get_llm_client()
        self.max_iterations = max_iterations

    def generate_scala_code(
        self,
        input_columns: list[str],
        target_columns: list[str],
        transformation_logic: dict[str, str],
        sample_records: list[dict] | None = None,
        progress_callback=None,
        crosswalk_tables: list[dict] | None = None,
    ) -> dict:
        """
        Generate Scala code for each target column transformation.

        Args:
            input_columns: List of input column names.
            target_columns: List of target column names.
            transformation_logic: Dict mapping target_col -> English description of how to derive it.
            sample_records: Optional list of dicts representing sample input rows.
            progress_callback: Optional callback(current, total, message).
            crosswalk_tables: Optional list of dicts with crosswalk/lookup table info.
                Each dict has: name, columns, sample_data

        Returns:
            dict with keys:
                - 'per_column_code': dict mapping target_col -> Scala code snippet
                - 'full_code': Complete Scala object with all transformations
                - 'sample_output': list of dicts with transformed sample records (if sample_records provided)
                - 'iterations_used': int
        """
        print(f"DEBUG: Starting Scala code generation")
        print(f"DEBUG: Input columns: {input_columns}")
        print(f"DEBUG: Target columns: {target_columns}")
        print(f"DEBUG: Transformation logic keys: {list(transformation_logic.keys())}")
        print(f"DEBUG: Sample records provided: {sample_records is not None}")
        print(f"DEBUG: Crosswalk tables provided: {crosswalk_tables is not None}")

        if progress_callback:
            progress_callback(0, self.max_iterations + 1, "Building prompt...")

        prompt = self._build_prompt(input_columns, target_columns, transformation_logic, sample_records, crosswalk_tables)

        if progress_callback:
            progress_callback(1, self.max_iterations + 1, "Generating Scala code with AI...")

        iteration = 0
        last_error = None
        response_text = ""

        while iteration < self.max_iterations:
            iteration += 1
            print(f"DEBUG: Generation iteration {iteration}")

            if progress_callback:
                progress_callback(
                    iteration,
                    self.max_iterations + 1,
                    f"AI generation attempt {iteration}...",
                )

            try:
                if iteration == 1:
                    current_prompt = prompt
                else:
                    current_prompt = self._build_refinement_prompt(
                        prompt, response_text, last_error
                    )

                response = self.llm_client.invoke([HumanMessage(content=current_prompt)])
                response_text = (
                    str(response.content) if hasattr(response, "content") else str(response)
                )

                result = self._parse_response(
                    response_text, target_columns, sample_records
                )

                if progress_callback:
                    progress_callback(
                        self.max_iterations + 1,
                        self.max_iterations + 1,
                        "Scala code generated successfully!",
                    )

                result["iterations_used"] = iteration
                return result

            except Exception as e:
                last_error = str(e)
                print(f"DEBUG: Iteration {iteration} failed: {last_error}")

                if iteration >= self.max_iterations:
                    if progress_callback:
                        progress_callback(
                            self.max_iterations + 1,
                            self.max_iterations + 1,
                            f"Returning best attempt after {iteration} iterations.",
                        )
                    # Return whatever we have
                    return {
                        "per_column_code": {},
                        "full_code": response_text if response_text else f"// Generation failed: {last_error}",
                        "sample_output": [],
                        "iterations_used": iteration,
                        "error": last_error,
                    }

        return {
            "per_column_code": {},
            "full_code": "// No code generated",
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
        crosswalk_tables: list[dict] | None = None,
    ) -> str:
        prompt = f"""You are an expert Scala/Spark developer. Generate clean, production-ready Scala code that transforms input data columns into target columns.

=== INPUT COLUMNS (Source DataFrame) ===
{json.dumps(input_columns, indent=2)}

=== TARGET COLUMNS (Output DataFrame) ===
{json.dumps(target_columns, indent=2)}

=== TRANSFORMATION LOGIC (English) ===
"""
        for target_col in target_columns:
            logic = transformation_logic.get(target_col, "No transformation specified — pass through or leave empty.")
            prompt += f'  {target_col}: {logic}\n'

        # Add crosswalk/lookup tables if provided
        if crosswalk_tables:
            prompt += """
=== CROSSWALK / LOOKUP TABLES ===
The following reference tables are available for joins/lookups. Use them when the transformation logic mentions lookups, mappings, or references to these tables.

"""
            for cw in crosswalk_tables:
                prompt += f"""
**Table: {cw['name']}**
Columns: {json.dumps(cw['columns'])}
Sample Data:
{json.dumps(cw['sample_data'], indent=2)}
"""
            prompt += """
CROSSWALK USAGE INSTRUCTIONS:
- When transformation logic mentions joining with a crosswalk table, use Spark DataFrame join operations.
- Assume each crosswalk table is available as a DataFrame with the same name (e.g., `gender_lookup`, `country_codes`).
- Use broadcast joins for small lookup tables: `df.join(broadcast(lookupDf), joinCondition)`
- Handle null values gracefully when lookups don't match.
"""

        if sample_records:
            prompt += f"""
=== SAMPLE INPUT RECORDS ===
{json.dumps(sample_records, indent=2)}
"""

        prompt += """
=== INSTRUCTIONS ===
Generate your response in EXACTLY this JSON structure (no extra text outside the JSON):

```json
{
  "per_column_code": {
    "<target_col_1>": "<Scala expression or withColumn snippet for this column>",
    "<target_col_2>": "<Scala expression or withColumn snippet for this column>"
  },
  "full_code": "<Complete Scala object/function that applies ALL transformations using Spark DataFrame API. Include all necessary imports, crosswalk DataFrame parameters, and a transform function.>",
  "sample_output": [
    {"<target_col_1>": "<value>", "<target_col_2>": "<value>"},
    ...
  ]
}
```

RULES:
1. Use Apache Spark DataFrame API (`withColumn`, `col`, `lit`, `when`, `concat`, `split`, `join`, `broadcast`, etc.).
2. Each value in `per_column_code` should be a SINGLE `.withColumn(...)` call or a concise Scala expression.
3. `full_code` must be a complete, compilable Scala object with:
   - All necessary imports (`org.apache.spark.sql.functions._`, `org.apache.spark.sql.DataFrame`)
   - If crosswalk tables are used, include them as parameters to the transform function
   - A `def transform(df: DataFrame, ...): DataFrame` function that chains all transformations
4. If sample input records are provided, compute `sample_output` by mentally applying the transformations. If no sample records are provided, return an empty array for `sample_output`.
5. Handle nulls gracefully using `coalesce`, `when`, or `na.fill`.
6. Use proper Spark SQL functions — do NOT use UDFs unless absolutely necessary.
7. For crosswalk joins, use `broadcast()` for small lookup tables and handle non-matching rows.
8. Return ONLY the JSON block. No commentary before or after.
"""
        return prompt

    def _build_refinement_prompt(self, original_prompt: str, previous_response: str, error: str) -> str:
        return f"""{original_prompt}

=== PREVIOUS ATTEMPT ===
{previous_response[:3000]}

=== ERROR ===
{error}

Please fix the issues and return the corrected JSON response. Return ONLY the JSON block."""

    # ------------------------------------------------------------------
    # Response parser
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        response_text: str,
        target_columns: list[str],
        sample_records: list[dict] | None,
    ) -> dict:
        """Parse the LLM response and extract structured data."""
        print(f"DEBUG: Parsing response (length: {len(response_text)})")

        # Try to extract JSON from the response
        json_text = response_text

        # Strip markdown code fences if present
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0].strip()
        elif "```" in json_text:
            json_text = json_text.split("```")[1].split("```")[0].strip()

        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            match = re.search(r'\{[\s\S]*\}', json_text)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError as e:
                    raise ValueError(f"Could not parse JSON from LLM response: {e}")
            else:
                raise ValueError("No JSON object found in LLM response.")

        per_column_code = parsed.get("per_column_code", {})
        full_code = parsed.get("full_code", "")
        sample_output = parsed.get("sample_output", [])

        # Unescape the full_code if it's a JSON-encoded string with \\n etc.
        if isinstance(full_code, str):
            full_code = full_code.replace("\\n", "\n").replace("\\t", "\t")

        return {
            "per_column_code": per_column_code,
            "full_code": full_code,
            "sample_output": sample_output,
        }
