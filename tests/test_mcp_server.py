"""Focused tests for the lifecycle-managed internal MCP tool session."""

import pytest

from mcp_layer.client import (
    EXPECTED_TOOLS,
    InternalMCPClient,
)
from models.schemas import (
    AnalysisPlan,
    IntentResult,
    PlanStep,
)


def build_query_arguments(sql: str) -> dict[str, str]:
    intent = IntentResult(
        original_question="Show the top 5 products by revenue.",
        rewritten_question="Rank five products by revenue.",
        analysis_type="ranking",
        metrics=["revenue"],
        dimensions=["product"],
        sort_order="descending",
        limit=5,
    )
    plan = AnalysisPlan(
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
    return {
        "sql": sql,
        "intent_json": intent.model_dump_json(),
        "plan_json": plan.model_dump_json(),
    }


def test_internal_mcp_session_exposes_only_approved_tools():
    with InternalMCPClient() as client:
        assert client.tool_names == frozenset(EXPECTED_TOOLS)
        schema = client.call_tool(
            "get_approved_schema",
            {},
        )

    assert "schema" in schema
    assert "Orders" in schema["schema"]
    assert client.tool_names == frozenset()


def test_mcp_executes_read_only_query_and_rejects_mutation():
    with InternalMCPClient() as client:
        result = client.call_tool(
            "execute_validated_read_query",
            build_query_arguments(
                'SELECT p.ProductName AS product, '
                'SUM(od.UnitPrice * od.Quantity) AS revenue '
                'FROM Products AS p '
                'JOIN "Order Details" AS od '
                'ON p.ProductID = od.ProductID '
                'GROUP BY p.ProductName '
                'ORDER BY revenue DESC LIMIT 5'
            ),
        )
        rejected = client.call_tool(
            "execute_validated_read_query",
            build_query_arguments("DROP TABLE Orders"),
        )

    assert result["success"] is True
    assert result["query_result"]["row_count"] == 5
    assert rejected["success"] is False
    assert rejected["stage"] == "structural_validation"


def test_mcp_tool_does_not_accept_caller_selected_role():
    with InternalMCPClient() as client:
        properties = client.tool_input_schemas[
            "execute_validated_read_query"
        ]["properties"]

    with pytest.raises(ValueError):
        client.call_tool("execute_read_sql", {"sql": "SELECT 1"})
    assert set(properties) == {"sql", "intent_json", "plan_json"}
