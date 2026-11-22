"""
Intent Understanding Agent.

This agent converts a natural-language business question into a
validated IntentResult.

Model:
    llama3.2:latest running locally through Ollama

Responsibilities:
    1. Understand the user's analytical question.
    2. Extract metrics, dimensions, filters, and time requirements.
    3. Identify the requested analysis type.
    4. Determine whether clarification is necessary.
    5. Return schema-validated structured output.

This agent does not:
    - Inspect the database schema.
    - Generate SQL.
    - Execute SQL.
    - Calculate analytical results.
    - Apply hardcoded business rules.
"""

from ollama import Client
from pydantic import ValidationError

from models.schemas import (
    IntentResult,
    TimePeriod,
)


class IntentAgent:
    """
    Model-driven Intent Understanding Agent.

    The agent uses Ollama structured output and validates the response
    using the shared IntentResult Pydantic model.
    """

    def __init__(
        self,
        model: str = "llama3.2:latest",
        host: str = "http://localhost:11434",
    ) -> None:
        """
        Initialize the Intent Agent.

        Args:
            model:
                Name of the Ollama model used for intent extraction.

            host:
                Address of the local Ollama API.
        """

        if not model.strip():
            raise ValueError(
                "The Ollama model name cannot be empty."
            )

        if not host.strip():
            raise ValueError(
                "The Ollama host cannot be empty."
            )

        self.model = model.strip()
        self.host = host.rstrip("/")
        self.client = Client(host=self.host)

    def _build_system_prompt(self) -> str:
        """
        Build the model instructions for intent extraction.

        The instructions describe the required reasoning principles but
        do not contain hardcoded database tables, columns, or mappings.
        """

        return """
You are the Intent Understanding Agent in a multi-agent business
data analytics system.

Your only responsibility is to understand the user's analytical
question and return structured analytical intent.

Determine the intent from the meaning of the complete question.

Extract:

1. The original question.
2. A clearer analytical version of the question.
3. The primary analysis type.
4. Requested business metrics.
5. Requested dimensions or grouping concepts.
6. Explicit filter conditions.
7. Time-period requirements.
8. Sorting requirements, when relevant.
9. Result limits, when explicitly requested.
10. Whether clarification is required.

Analysis-type meanings:

- aggregation:
  Calculate totals, counts, averages, minimums, maximums, or other
  summarized measures.

- comparison:
  Compare two or more periods, categories, groups, or segments.

- trend_analysis:
  Analyze how a measure changes across time.

- ranking:
  Order entities based on a metric and identify leading or trailing
  results.

- root_cause_analysis:
  Investigate possible reasons or contributors behind an observed
  outcome.

- anomaly_detection:
  Identify unusual observations, spikes, drops, or outliers.

- distribution_analysis:
  Analyze how values are distributed across a population or range.

- general_analysis:
  Use only when no more specific analysis type is supported by the
  question.

Rules:

- Use semantic understanding of the full question.
- Do not generate SQL.
- Do not answer the business question.
- Do not calculate results.
- Do not invent database tables or columns.
- Do not assume a specific database schema.
- Do not invent filters, dates, or business measures.
- Preserve explicit filter values from the question.
- Express filters using field, operator, and value.
- Use lowercase, concise names for metrics and dimensions.
- Extract a limit only when the user requests a specific number.
- Extract a sort order only when ordering is relevant to the request.
- Determine the sorting direction from the requested ranking objective.
- A ranking that asks for leading, largest, greatest, highest, or
  best-performing values requires descending order.
- A ranking that asks for trailing, smallest, least, lowest, or
  worst-performing values requires ascending order.
- Confirm that the selected sorting direction would return the
  requested records before producing the final structured output.
- Use ISO date format when exact dates can be represented reliably.
- Otherwise preserve the user's business-period expression.
- Ask for clarification only when missing information prevents a
  meaningful analytical request from being formed.
- If clarification_required is false, clarification_question must
  be null.
- If clarification_required is true, clarification_question must
  contain one focused question.
- Return only JSON matching the supplied schema.
""".strip()

    def _build_user_prompt(
        self,
        user_question: str,
    ) -> str:
        """
        Build the user message sent to the model.
        """

        return f"""
Analyze the following business question and return its structured
analytical intent.

Business question:
{user_question}

Return only the structured result matching the supplied JSON schema.
Do not include Markdown, commentary, SQL, or an analytical answer.
""".strip()

    def _call_model(
        self,
        user_question: str,
    ) -> str:
        """
        Call the local Ollama model and return its raw response.

        Args:
            user_question:
                Cleaned business question.

        Returns:
            Raw JSON text produced by the model.
        """

        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._build_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": self._build_user_prompt(
                            user_question
                        ),
                    },
                ],
                format=IntentResult.model_json_schema(),
                stream=False,
                options={
                    "temperature": 0,
                    "seed": 42,
                },
            )

        except Exception as exc:
            raise ConnectionError(
                "Could not communicate with Ollama.\n"
                f"Host: {self.host}\n"
                f"Model: {self.model}\n"
                "Make sure Ollama is running and the model is "
                f"installed.\nOriginal error: {exc}"
            ) from exc

        raw_content = response.message.content

        if raw_content is None:
            raise RuntimeError(
                "The Intent Agent returned no content."
            )

        cleaned_content = raw_content.strip()

        if not cleaned_content:
            raise RuntimeError(
                "The Intent Agent returned an empty response."
            )

        return cleaned_content

    def _validate_response(
        self,
        raw_content: str,
    ) -> IntentResult:
        """
        Validate the model response using IntentResult.

        No rule-based correction is applied. Invalid output is rejected
        instead of being silently modified.
        """

        try:
            return IntentResult.model_validate_json(
                raw_content
            )

        except ValidationError as exc:
            raise RuntimeError(
                "The Intent Agent returned output that did not match "
                "the required IntentResult schema.\n\n"
                f"Raw model output:\n{raw_content}\n\n"
                f"Validation error:\n{exc}"
            ) from exc

    def understand(
        self,
        user_question: str,
    ) -> IntentResult:
        """
        Convert a natural-language analytical question into a
        validated IntentResult.

        Args:
            user_question:
                Business question entered by the user.

        Returns:
            Validated IntentResult.

        Raises:
            TypeError:
                If user_question is not a string.

            ValueError:
                If user_question is empty.

            ConnectionError:
                If Ollama cannot be reached.

            RuntimeError:
                If the model returns empty or invalid structured output.
        """

        if not isinstance(user_question, str):
            raise TypeError(
                "user_question must be a string."
            )

        cleaned_question = user_question.strip()

        if not cleaned_question:
            raise ValueError(
                "The user question cannot be empty."
            )

        normalized_question = (
            cleaned_question.lower().strip()
            .rstrip(".!?")
        )
        incomplete_phrases = {
            "analyze",
            "analyze the results",
            "analyse",
            "analyse the results",
            "give me insights",
            "show me the data",
        }

        if normalized_question in incomplete_phrases:
            return IntentResult(
                original_question=cleaned_question,
                rewritten_question=cleaned_question,
                analysis_type="general_analysis",
                metrics=[],
                dimensions=[],
                filters=[],
                time_period=TimePeriod(),
                sort_order=None,
                limit=None,
                clarification_required=True,
                clarification_question=(
                    "Which metric and dimension would you like "
                    "me to analyze?"
                ),
            )

        raw_content = self._call_model(
            user_question=cleaned_question
        )

        result = self._validate_response(
            raw_content=raw_content
        )

        return result


def main() -> None:
    """
    Run a simple standalone Intent Agent smoke test.

    A separate test file performs broader validation.
    """

    print("=" * 80)
    print("INTENT AGENT SMOKE TEST")
    print("=" * 80)

    question = (
        "Show the top 5 products by revenue in 2025."
    )

    print(f"Model    : llama3.2:latest")
    print(f"Question : {question}")
    print("\nCalling local Ollama model...")

    try:
        agent = IntentAgent(
            model="llama3.2:latest",
            host="http://localhost:11434",
        )

        result = agent.understand(question)

        print("\nStructured Intent:")
        print(result.model_dump_json(indent=2))

        print("\n" + "=" * 80)
        print("RESULT: INTENT AGENT IS WORKING")
        print("=" * 80)

    except Exception as exc:
        print("\n" + "=" * 80)
        print("RESULT: INTENT AGENT FAILED")
        print("=" * 80)
        print(f"{type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()