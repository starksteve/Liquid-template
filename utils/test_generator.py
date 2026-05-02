from utils.llm_config import get_llm_client
from langchain_core.messages import HumanMessage
import json
import re


class TestGenerator:
    """
    Generate unit tests for produced Scala and PySpark code using Azure OpenAI.
    - ScalaTest (AnyFlatSpec) for Scala/Spark code
    - pytest + pyspark for PySpark code
    """

    def __init__(self):
        self.llm_client = get_llm_client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_tests(
        self,
        input_columns: list[str],
        target_columns: list[str],
        transformation_logic: dict[str, str],
        scala_code: str | None = None,
        pyspark_code: str | None = None,
        sample_records: list[dict] | None = None,
        progress_callback=None,
    ) -> dict:
        """
        Generate unit tests for the provided code artifacts.

        Returns dict:
            - 'scala_tests': str  — ScalaTest source file
            - 'pytest_tests': str — pytest source file
        """
        results: dict = {}

        if progress_callback:
            progress_callback(0, 4, "Preparing test generation…")

        if scala_code:
            if progress_callback:
                progress_callback(1, 4, "Generating ScalaTest suite…")
            results["scala_tests"] = self._generate_scala_tests(
                input_columns, target_columns, transformation_logic,
                scala_code, sample_records
            )
        else:
            results["scala_tests"] = ""

        if pyspark_code:
            if progress_callback:
                progress_callback(3, 4, "Generating pytest suite…")
            results["pytest_tests"] = self._generate_pytest(
                input_columns, target_columns, transformation_logic,
                pyspark_code, sample_records
            )
        else:
            results["pytest_tests"] = ""

        if progress_callback:
            progress_callback(4, 4, "Tests ready!")

        return results

    # ------------------------------------------------------------------
    # Scala tests
    # ------------------------------------------------------------------

    def _generate_scala_tests(
        self,
        input_columns: list[str],
        target_columns: list[str],
        transformation_logic: dict[str, str],
        scala_code: str,
        sample_records: list[dict] | None,
    ) -> str:
        prompt = f"""You are an expert in Scala, Apache Spark, and ScalaTest. Write a complete unit test file for the transformation code below.

=== SCALA TRANSFORMATION CODE ===
```scala
{scala_code[:4000]}
```

=== TRANSFORMATION LOGIC ===
"""
        for col in target_columns:
            logic = transformation_logic.get(col, "")
            if logic:
                prompt += f"  {col}: {logic}\n"

        prompt += f"""
=== INPUT COLUMNS ===
{json.dumps(input_columns)}

=== TARGET COLUMNS ===
{json.dumps(target_columns)}
"""
        if sample_records:
            prompt += f"\n=== SAMPLE INPUT RECORDS ===\n{json.dumps(sample_records[:3], indent=2)}\n"

        prompt += """
=== REQUIREMENTS ===
1. Use ScalaTest `AnyFlatSpec` + `Matchers` + `BeforeAndAfterAll`.
2. Use `SparkSession.builder.master("local[*]").appName("TransformTest").getOrCreate()`.
3. Write at least 5 test cases covering:
   a. Happy path — all columns present and valid
   b. Null/empty input values — should not throw exceptions
   c. Boundary values — min/max lengths, edge dates
   d. Crosswalk lookup miss — unmatched keys should return null/default
   e. Schema check — output DataFrame must contain all target columns
4. Each `it should ...` block must call the transform function and assert on specific column values using `===`.
5. Clean up SparkSession in `afterAll()`.
6. Add a `// TODO:` comment for any test cases a human reviewer should expand.
7. Return ONLY the Scala test file. No extra prose.
"""
        try:
            response = self.llm_client.invoke([HumanMessage(content=prompt)])
            raw = str(response.content) if hasattr(response, "content") else str(response)
            return self._extract_code(raw, "scala")
        except Exception as e:
            return f"// ScalaTest generation failed: {e}"

    # ------------------------------------------------------------------
    # pytest tests
    # ------------------------------------------------------------------

    def _generate_pytest(
        self,
        input_columns: list[str],
        target_columns: list[str],
        transformation_logic: dict[str, str],
        pyspark_code: str,
        sample_records: list[dict] | None,
    ) -> str:
        prompt = f"""You are an expert in PySpark and pytest. Write a complete pytest test file for the transformation code below.

=== PYSPARK TRANSFORMATION CODE ===
```python
{pyspark_code[:4000]}
```

=== TRANSFORMATION LOGIC ===
"""
        for col in target_columns:
            logic = transformation_logic.get(col, "")
            if logic:
                prompt += f"  {col}: {logic}\n"

        prompt += f"""
=== INPUT COLUMNS ===
{json.dumps(input_columns)}

=== TARGET COLUMNS ===
{json.dumps(target_columns)}
"""
        if sample_records:
            prompt += f"\n=== SAMPLE INPUT RECORDS ===\n{json.dumps(sample_records[:3], indent=2)}\n"

        prompt += """
=== REQUIREMENTS ===
1. Use `pytest` with `pyspark` installed.
2. Create a `@pytest.fixture(scope="session")` for SparkSession with `master="local[*]"`.
3. Write at least 5 test functions covering:
   a. Happy path — verify specific column values after transform
   b. Null input handling — input rows with None values must not raise errors
   c. Schema validation — output must have exactly the target columns
   d. Row count preservation — input and output row counts must match
   e. Data type checks — numeric/date columns should have correct dtypes
4. Use `spark.createDataFrame([...], schema)` to build test DataFrames.
5. Use `assert df.filter(F.col("col") == expected).count() == 1` style assertions.
6. Import the `transform` function from the module under test.
7. Add `# TODO:` comments for any cases a reviewer should expand.
8. Return ONLY the Python test file. No extra prose.
"""
        try:
            response = self.llm_client.invoke([HumanMessage(content=prompt)])
            raw = str(response.content) if hasattr(response, "content") else str(response)
            return self._extract_code(raw, "python")
        except Exception as e:
            return f"# pytest generation failed: {e}"

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _extract_code(self, text: str, lang: str) -> str:
        pattern = rf"```{lang}\s*(.*?)```"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # fallback: strip any fences
        text = re.sub(r"```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```", "", text)
        return text.strip()
