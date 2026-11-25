"""Focused tests for SQL alignment checks before SQLite execution."""

from types import SimpleNamespace

from database.db_manager import DatabaseManager
from agents.sql_agent import SQLAgent
from models.schemas import (
    AnalysisPlan,
    FilterCondition,
    IntentResult,
    PlanStep,
    TimePeriod,
)
from security.schema_policy import SchemaPolicyValidator
from security.models import UserRole
from tools.sql_execution_tool import SQLExecutionTool
from tools.sql_validator import SQLValidator


VALID_SQL = """
SELECT p.ProductName AS product,
       SUM(od.UnitPrice * od.Quantity) AS revenue
FROM Products AS p
JOIN "Order Details" AS od ON p.ProductID = od.ProductID
GROUP BY p.ProductName
ORDER BY revenue DESC
LIMIT 5
"""


def build_intent(
    filters: list[FilterCondition] | None = None,
) -> IntentResult:
    return IntentResult(
        original_question="Show the top 5 products by revenue.",
        rewritten_question="Rank five products by revenue.",
        analysis_type="ranking",
        metrics=["revenue"],
        dimensions=["product"],
        filters=filters or [],
        sort_order="descending",
        limit=5,
    )


def build_plan() -> AnalysisPlan:
    return AnalysisPlan(
        objective="Rank products by revenue.",
        strategy="Aggregate and rank product revenue.",
        steps=[
            PlanStep(
                step_id=1,
                title="Calculate product revenue",
                description="Aggregate revenue by product.",
                step_type="ranking",
                required_metrics=["revenue"],
                required_dimensions=["product"],
                requires_sql=True,
                expected_output="Five products ordered by revenue.",
            )
        ],
    )


def build_tool() -> SQLExecutionTool:
    database_manager = DatabaseManager(max_rows=100)
    validator = SQLValidator(dialect="sqlite")
    return SQLExecutionTool(
        database_manager=database_manager,
        sql_validator=validator,
        schema_policy_validator=SchemaPolicyValidator(
            database_manager=database_manager,
            sql_validator=validator,
            maximum_result_rows=100,
        ),
        role=UserRole.ANALYST,
        explain_before_execution=True,
    )


def test_aligned_query_passes_business_validation_and_executes():
    outcome = build_tool().validate_and_execute(
        VALID_SQL,
        intent=build_intent(),
        plan=build_plan(),
    )

    assert outcome.success is True
    assert outcome.query_result is not None
    assert outcome.query_result.columns == ["product", "revenue"]


def test_query_exceeding_requested_limit_is_retryable_before_execution():
    outcome = build_tool().validate_and_execute(
        VALID_SQL.replace("LIMIT 5", "LIMIT 8"),
        intent=build_intent(),
        plan=build_plan(),
    )

    assert outcome.success is False
    assert outcome.stage == "intent_alignment"
    assert outcome.retryable is True
    assert any("requested maximum" in error for error in outcome.validation_errors)


def test_query_must_apply_explicit_filter_and_operator():
    intent = build_intent(
        filters=[
            FilterCondition(
                field="country",
                operator="equals",
                value="USA",
            )
        ]
    )
    outcome = build_tool().validate_and_execute(
        VALID_SQL,
        intent=intent,
        plan=build_plan(),
    )

    assert outcome.success is False
    assert outcome.stage == "intent_alignment"
    assert any("filter" in error.lower() for error in outcome.validation_errors)


def test_monthly_trend_does_not_require_a_date_filter():
    intent = IntentResult(
        original_question="Show monthly revenue trend.",
        rewritten_question="Show monthly revenue over time.",
        analysis_type="trend_analysis",
        metrics=["revenue"],
        dimensions=["month"],
        filters=[],
        time_period=TimePeriod(granularity="month"),
        sort_order="ascending",
        limit=None,
    )
    plan = AnalysisPlan(
        objective="Show monthly revenue trend.",
        strategy="Aggregate revenue by order month.",
        steps=[
            PlanStep(
                step_id=1,
                title="Aggregate monthly revenue",
                description="Group revenue by order month.",
                step_type="trend_analysis",
                required_metrics=["revenue"],
                required_dimensions=["month"],
                requires_sql=True,
                expected_output="Monthly revenue values in order.",
            )
        ],
    )
    sql = """
    SELECT strftime('%Y-%m', o.OrderDate) AS month,
           SUM(od.UnitPrice * od.Quantity) AS revenue
    FROM Orders AS o
    JOIN "Order Details" AS od ON o.OrderID = od.OrderID
    GROUP BY strftime('%Y-%m', o.OrderDate)
    ORDER BY month ASC
    """

    outcome = build_tool().validate_and_execute(
        sql,
        intent=intent,
        plan=plan,
    )

    assert outcome.success is True, outcome.validation_errors


def test_retry_uses_alignment_feedback_instead_of_repeating_schema_fallback():
    schema = """
TABLE: Products
- ProductID INTEGER
- ProductName TEXT
- CategoryID INTEGER
TABLE: Order Details
- OrderID INTEGER
- ProductID INTEGER
- UnitPrice REAL
- Quantity INTEGER
TABLE: Categories
- CategoryID INTEGER
- CategoryName TEXT
"""
    intent = build_intent()
    plan = AnalysisPlan(
        objective="Rank products by revenue and category.",
        strategy="Aggregate revenue by product and category.",
        steps=[
            PlanStep(
                step_id=1,
                title="Calculate product revenue by category",
                description="Group revenue by product and category.",
                step_type="ranking",
                required_metrics=["revenue"],
                required_dimensions=["product", "category"],
                requires_sql=True,
                expected_output="Products ranked by revenue within category.",
            )
        ],
    )
    corrected_sql = """
SELECT p.ProductName AS product,
       c.CategoryName AS category,
       SUM(od.UnitPrice * od.Quantity) AS revenue
FROM Products AS p
JOIN "Order Details" AS od ON p.ProductID = od.ProductID
JOIN Categories AS c ON p.CategoryID = c.CategoryID
GROUP BY p.ProductName, c.CategoryName
ORDER BY revenue DESC
LIMIT 5
"""
    agent = SQLAgent(max_attempts=1)
    prompts = []

    def generate_corrected_sql(prompt: str) -> str:
        prompts.append(prompt)
        return corrected_sql

    agent._call_model = generate_corrected_sql
    result = agent.generate_sql(
        user_question=intent.original_question,
        intent=intent,
        plan=plan,
        database_schema=schema,
        previous_sql=VALID_SQL,
        previous_error=(
            "The query does not include requested dimension 'category'."
        ),
    )

    assert prompts
    assert "requested dimension 'category'" in prompts[0]
    outcome = build_tool().validate_and_execute(
        result.sql,
        intent=intent,
        plan=plan,
    )
    assert outcome.success is True


def test_compact_retry_prompt_preserves_previous_sql():
    agent = SQLAgent(max_attempts=1)
    prompt = agent._generation_prompt(
        user_question="Show the top 5 products by revenue.",
        intent=build_intent(),
        plan=build_plan(),
        database_schema="TABLE: Products\n- ProductID INTEGER",
        previous_sql=VALID_SQL,
        previous_error="The previous metric did not match.",
    )

    compact_prompt = agent._compact_prompt(prompt)

    assert VALID_SQL.strip() in compact_prompt


def test_retry_prompt_uses_sqlcoder_before_fallback():
    agent = SQLAgent(max_attempts=1)
    calls = []

    class FakeClient:
        def generate(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(response="SELECT 1")

    agent.client = FakeClient()
    prompt = agent._generation_prompt(
        user_question="Show the top 5 products by revenue.",
        intent=build_intent(),
        plan=build_plan(),
        database_schema="TABLE: Products\n- ProductID INTEGER",
        previous_sql=VALID_SQL,
        previous_error=(
            "The query does not include requested dimension 'category'."
        ),
    )

    result = agent._call_model(prompt)

    assert result == "SELECT 1"
    assert len(calls) == 1
    assert calls[0]["model"] == "sqlcoder:latest"
    assert VALID_SQL.strip() in calls[0]["prompt"]
    assert "requested dimension 'category'" in calls[0]["prompt"]
