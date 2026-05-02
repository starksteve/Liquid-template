"""
quality_rules_generator.py
---------------------------
Generates Spark data quality validation code based on:
- Source column profiles (nulls, types, ranges, patterns)
- User-defined expectations

Produces a runnable Spark function that validates data BEFORE
the transformation runs — a real Senior DE pattern.
"""

from utils.llm_config import get_llm_client
from langchain_core.messages import HumanMessage
import json
import re
import pandas as pd


class QualityRulesGenerator:
    """
    AI-powered data quality rules generator.
    Outputs PySpark + Scala validation code.
    """

    def __init__(self):
        self.llm_client = get_llm_client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        source_df: pd.DataFrame,
        input_columns: list[str],
        target_columns: list[str],
        transformation_logic: dict[str, str],
        progress_callback=None,
    ) -> dict:
        """
        Generate data quality validation code.

        Returns dict:
            - 'pyspark_dq': str   — PySpark validation function
            - 'scala_dq':   str   — Scala validation function
            - 'rules_summary': list[dict]  — human-readable rules list
        """
        if progress_callback:
            progress_callback(0, 2, "Analyzing data for quality rules…")

        # Build a lightweight profile for the prompt
        profile = self._quick_profile(source_df, input_columns)

        if progress_callback:
            progress_callback(1, 2, "Generating quality validation code…")

        prompt = self._build_prompt(
            input_columns, target_columns, transformation_logic, profile
        )

        try:
            response = self.llm_client.invoke([HumanMessage(content=prompt)])
            raw = str(response.content) if hasattr(response, "content") else str(response)
            result = self._parse_response(raw)
            if progress_callback:
                progress_callback(2, 2, "Quality rules ready!")
            return result
        except Exception as e:
            print(f"DEBUG: QualityRulesGenerator failed: {e}")
            return {
                "pyspark_dq": f"# Quality rules generation failed: {e}",
                "scala_dq": f"// Quality rules generation failed: {e}",
                "rules_summary": [],
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # Quick profiler (no LLM)
    # ------------------------------------------------------------------

    def _quick_profile(self, df: pd.DataFrame, columns: list[str]) -> dict:
        profile = {}
        for col in columns:
            if col not in df.columns:
                continue
            s = df[col]
            p = {
                "dtype": str(s.dtype),
                "null_pct": round(s.isnull().mean() * 100, 1),
                "unique_count": int(s.nunique()),
                "sample_values": [str(v) for v in s.dropna().head(5).tolist()],
            }
            if pd.api.types.is_numeric_dtype(s):
                clean = s.dropna()
                if len(clean) > 0:
                    p["min"] = float(clean.min())
                    p["max"] = float(clean.max())
            else:
                clean = s.dropna().astype(str)
                if len(clean) > 0:
                    p["max_length"] = int(clean.str.len().max())
            profile[col] = p
        return profile

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        input_columns: list[str],
        target_columns: list[str],
        transformation_logic: dict[str, str],
        profile: dict,
    ) -> str:
        prompt = f"""You are a Senior Data Engineer. Generate production-ready Spark data quality validation code.

The validation function runs BEFORE the main transformation — it checks the source data and raises a clear error if any quality gate fails.

=== SOURCE COLUMN PROFILES ===
{json.dumps(profile, indent=2)}

=== TRANSFORMATION LOGIC (what we're about to do to this data) ===
"""
        for col in input_columns:
            logic = transformation_logic.get(col, "")
            if logic:
                prompt += f"  {col}: {logic}\n"

        prompt += f"""
=== TARGET COLUMNS ===
{json.dumps(target_columns)}

=== RULES TO GENERATE ===
Based on the column profiles, generate sensible quality rules such as:
- NOT NULL checks for columns that have 0% nulls in the profile (likely required fields)
- Range checks for numeric columns (min/max bounds)
- Max length checks for string columns
- Referential integrity checks if columns look like foreign keys
- Pattern checks if values look like dates, emails, IDs, etc.
- Duplicate row check on likely key columns

=== REQUIRED RESPONSE FORMAT ===
Respond with ONLY this JSON:

```json
{{
  "rules_summary": [
    {{
      "column": "<column_name>",
      "rule": "<short rule name>",
      "description": "<plain English description>",
      "severity": "error"
    }}
  ],
  "pyspark_dq": "<Complete PySpark validation function as a string. Must: import pyspark.sql.functions as F, accept df: DataFrame, return DataFrame if all checks pass, raise ValueError with clear message if any check fails. Include a __main__ example.>",
  "scala_dq": "<Complete Scala validation function. Must: accept df: DataFrame, return DataFrame if valid, throw IllegalArgumentException with message if invalid.>"
}}
```

=== CODING STANDARDS ===
PySpark:
1. `def validate_data_quality(df: DataFrame) -> DataFrame:`
2. For each rule, compute the violation count: `df.filter(F.col(...).isNull()).count()`
3. Collect all violations first, then raise a single ValueError listing ALL failures.
4. Log each rule check with `print(f"DQ CHECK: rule_name ... PASS/FAIL (N violations)")`.
5. Return `df` unchanged if all checks pass.

Scala:
1. `def validateDataQuality(df: DataFrame): DataFrame`
2. Same pattern — collect all violations, throw once.
3. Use `df.filter(col(...).isNull).count()` style checks.

Return ONLY the JSON block.
"""
        return prompt

    # ------------------------------------------------------------------
    # Parser
    # ------------------------------------------------------------------

    def _parse_response(self, text: str) -> dict:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        raw = match.group(1) if match else text.strip()
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

        try:
            data = json.loads(raw)
            return {
                "pyspark_dq": data.get("pyspark_dq", "# Not generated"),
                "scala_dq": data.get("scala_dq", "// Not generated"),
                "rules_summary": data.get("rules_summary", []),
                "error": None,
            }
        except json.JSONDecodeError:
            # Try extracting code blocks directly
            py_match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
            scala_match = re.search(r"```scala\s*(.*?)```", text, re.DOTALL)
            return {
                "pyspark_dq": py_match.group(1).strip() if py_match else text,
                "scala_dq": scala_match.group(1).strip() if scala_match else "// Not generated",
                "rules_summary": [],
                "error": "Could not parse structured JSON — code blocks extracted directly.",
            }
