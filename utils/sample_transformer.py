from utils.llm_config import get_llm_client
from langchain_core.messages import HumanMessage
import json
import re


class SampleTransformer:
    """
    Apply transformation logic to sample input records and produce target records.
    Uses Azure OpenAI to interpret English transformation logic and compute outputs.
    """

    def __init__(self):
        self.llm_client = get_llm_client()

    def transform_samples(
        self,
        input_columns: list[str],
        target_columns: list[str],
        transformation_logic: dict[str, str],
        sample_records: list[dict],
    ) -> list[dict]:
        """
        Given sample input records and English transformation logic,
        produce transformed target records.

        Args:
            input_columns: List of input column names.
            target_columns: List of target column names.
            transformation_logic: Dict mapping target_col -> English description.
            sample_records: List of dicts, each dict is one input row.

        Returns:
            List of dicts — each dict is one target row with target column values.
        """
        if not sample_records:
            return []

        prompt = self._build_prompt(
            input_columns, target_columns, transformation_logic, sample_records
        )

        try:
            response = self.llm_client.invoke([HumanMessage(content=prompt)])
            response_text = (
                str(response.content) if hasattr(response, "content") else str(response)
            )
            return self._parse_response(response_text, target_columns)
        except Exception as e:
            print(f"DEBUG: Sample transformation failed: {e}")
            return []

    def _build_prompt(
        self,
        input_columns: list[str],
        target_columns: list[str],
        transformation_logic: dict[str, str],
        sample_records: list[dict],
    ) -> str:
        prompt = f"""You are a precise data transformation engine. Apply the transformation logic below to each input record and return the transformed target records.

=== INPUT COLUMNS ===
{json.dumps(input_columns)}

=== TARGET COLUMNS ===
{json.dumps(target_columns)}

=== TRANSFORMATION LOGIC ===
"""
        for target_col in target_columns:
            logic = transformation_logic.get(target_col, "Pass through as-is or leave empty.")
            prompt += f"  {target_col}: {logic}\n"

        prompt += f"""
=== SAMPLE INPUT RECORDS ===
{json.dumps(sample_records, indent=2)}

=== INSTRUCTIONS ===
Apply the transformation logic to each input record and produce corresponding target records.
Today's date for any date calculations: 2026-03-17

Return ONLY a JSON array of objects. Each object has the target column names as keys.
Example:
```json
[
  {{"target_col_1": "value1", "target_col_2": "value2"}},
  {{"target_col_1": "value3", "target_col_2": "value4"}}
]
```

RULES:
1. Apply transformations exactly as described.
2. One output record per input record, in the same order.
3. Return ONLY the JSON array, no extra text.
"""
        return prompt

    def _parse_response(self, response_text: str, target_columns: list[str]) -> list[dict]:
        """Parse the LLM response to extract the list of transformed records."""
        text = response_text.strip()

        # Strip markdown fences
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Try to find a JSON array
            match = re.search(r'\[[\s\S]*\]', text)
            if match:
                parsed = json.loads(match.group())
            else:
                raise ValueError("Could not parse transformed records from LLM response.")

        if not isinstance(parsed, list):
            parsed = [parsed]

        return parsed
