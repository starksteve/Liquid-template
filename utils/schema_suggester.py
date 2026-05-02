"""
schema_suggester.py
-------------------
Given a source CSV (and optionally a target CSV), uses the LLM to
automatically suggest transformation logic for each target column.

This removes the manual step of writing transformation logic —
the AI reads the actual data and figures out the mapping.
"""

from utils.llm_config import get_llm_client
from langchain_core.messages import HumanMessage
import json
import re
import pandas as pd


class SchemaSuggester:
    """
    AI-powered schema mapper.
    Reads source (and optionally target) CSV samples and
    suggests transformation logic for every target column.
    """

    def __init__(self):
        self.llm_client = get_llm_client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suggest(
        self,
        source_df: pd.DataFrame,
        target_columns: list[str],
        target_df: pd.DataFrame | None = None,
        extra_context: str = "",
    ) -> dict:
        """
        Suggest transformation logic for each target column.

        Args:
            source_df:      Source DataFrame (uploaded CSV).
            target_columns: List of desired output column names.
            target_df:      Optional target DataFrame — if the user also
                            uploads a sample of what the output should look like.
            extra_context:  Any free-text hints from the user.

        Returns:
            dict with keys:
                - 'suggestions': dict  target_col -> suggested logic string
                - 'confidence':  dict  target_col -> "high" | "medium" | "low"
                - 'notes':       list[str]  general observations about the mapping
        """
        prompt = self._build_prompt(source_df, target_columns, target_df, extra_context)

        try:
            response = self.llm_client.invoke([HumanMessage(content=prompt)])
            raw = str(response.content) if hasattr(response, "content") else str(response)
            return self._parse_response(raw, target_columns)
        except Exception as e:
            print(f"DEBUG: SchemaSuggester failed: {e}")
            return {
                "suggestions": {col: "" for col in target_columns},
                "confidence": {col: "low" for col in target_columns},
                "notes": [f"AI suggestion failed: {e}"],
            }

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        source_df: pd.DataFrame,
        target_columns: list[str],
        target_df: pd.DataFrame | None,
        extra_context: str,
    ) -> str:
        source_cols = list(source_df.columns)
        source_sample = source_df.head(5).to_dict(orient="records")
        source_dtypes = {col: str(dtype) for col, dtype in source_df.dtypes.items()}

        prompt = f"""You are a Senior Data Engineer. Your job is to analyze source data and suggest transformation logic to produce a target schema.

=== SOURCE SCHEMA ===
Columns: {json.dumps(source_cols)}
Data types: {json.dumps(source_dtypes)}
Sample rows:
{json.dumps(source_sample, indent=2, default=str)}
"""

        if target_df is not None:
            target_sample = target_df.head(5).to_dict(orient="records")
            target_dtypes = {col: str(dtype) for col, dtype in target_df.dtypes.items()}
            prompt += f"""
=== TARGET SCHEMA (example output provided) ===
Columns: {json.dumps(list(target_df.columns))}
Data types: {json.dumps(target_dtypes)}
Sample rows:
{json.dumps(target_sample, indent=2, default=str)}
"""

        prompt += f"""
=== DESIRED TARGET COLUMNS ===
{json.dumps(target_columns)}
"""

        if extra_context:
            prompt += f"\n=== ADDITIONAL CONTEXT FROM USER ===\n{extra_context}\n"

        prompt += """
=== YOUR TASK ===
For EACH target column, suggest:
1. A clear, concise English description of how to derive it from the source columns.
2. A confidence level: "high" (obvious mapping), "medium" (reasonable guess), or "low" (unclear).

Think like a Senior DE:
- Look for direct column name matches or near-matches.
- Look at sample values to infer lookups, concatenations, date formatting, type casts, etc.
- If a target column can't be derived from source, say so clearly.
- Be specific: don't say "transform the column" — say exactly what operation to apply.

=== REQUIRED RESPONSE FORMAT ===
Respond with ONLY this JSON (no prose outside):

```json
{
  "suggestions": {
    "<target_col_1>": "<plain English transformation description>",
    "<target_col_2>": "<plain English transformation description>"
  },
  "confidence": {
    "<target_col_1>": "high",
    "<target_col_2>": "medium"
  },
  "notes": [
    "<general observation about this mapping>",
    "<any warnings or assumptions made>"
  ]
}
```
"""
        return prompt

    # ------------------------------------------------------------------
    # Parser
    # ------------------------------------------------------------------

    def _parse_response(self, text: str, target_columns: list[str]) -> dict:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        raw = match.group(1) if match else text.strip()
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

        try:
            data = json.loads(raw)
            # Ensure all target columns are present
            suggestions = data.get("suggestions", {})
            confidence = data.get("confidence", {})
            for col in target_columns:
                if col not in suggestions:
                    suggestions[col] = ""
                if col not in confidence:
                    confidence[col] = "low"
            return {
                "suggestions": suggestions,
                "confidence": confidence,
                "notes": data.get("notes", []),
            }
        except json.JSONDecodeError:
            # Return raw text as a single note
            return {
                "suggestions": {col: "" for col in target_columns},
                "confidence": {col: "low" for col in target_columns},
                "notes": [f"Could not parse AI response. Raw output: {text[:500]}"],
            }
