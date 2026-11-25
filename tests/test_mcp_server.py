"""
Focused tests for MCP server discovery and safe database tools.
"""

import asyncio
from types import SimpleNamespace

from mcp_server import (
    _build_safe_state_summary,
    create_mcp_server,
)


def test_mcp_server_exposes_expected_tools():
    server = create_mcp_server()
    tools = asyncio.run(server.list_tools())
    names = {tool.name for tool in tools}

    assert {
        "list_database_objects",
        "get_table_columns",
        "execute_read_sql",
        "run_analytics",
    }.issubset(names)


def test_mcp_read_sql_rejects_non_read_query():
    server = create_mcp_server()
    result = asyncio.run(
        server.call_tool(
            "execute_read_sql",
            {"sql": "DROP TABLE Orders"},
        )
    )

    assert isinstance(result, tuple)
    assert len(result) == 2
    payload = result[1]
    assert isinstance(payload, dict)

    assert payload["success"] is False
    assert payload["stage"] == "structural_validation"
    assert "validated_sql" not in payload
    assert "execution_error" not in payload


def test_execute_read_sql_schema_does_not_expose_role():
    server = create_mcp_server()
    tools = asyncio.run(server.list_tools())
    sql_tool = next(
        tool for tool in tools
        if tool.name == "execute_read_sql"
    )

    assert set(sql_tool.inputSchema["properties"]) == {"sql"}
    assert "role" not in sql_tool.inputSchema["properties"]


def test_execute_read_sql_accepts_select_and_rejects_destructive_sql():
    server = create_mcp_server()

    select_payload = asyncio.run(
        server.call_tool(
            "execute_read_sql",
            {"sql": "SELECT OrderID FROM Orders LIMIT 1"},
        )
    )[1]
    assert select_payload["success"] is True

    for sql in (
        "DELETE FROM Orders",
        "UPDATE Orders SET ShipName=1",
        "INSERT INTO Orders(OrderID) VALUES(999999)",
        "DROP TABLE Orders",
        "CREATE TABLE bad(x)",
    ):
        payload = asyncio.run(
            server.call_tool(
                "execute_read_sql",
                {"sql": sql},
            )
        )[1]
        assert payload["success"] is False


def test_mcp_resources_and_prompt_are_discoverable():
    server = create_mcp_server()

    resources = asyncio.run(server.list_resources())
    resource_names = {str(resource.uri) for resource in resources}
    assert "northwind://schema" in resource_names
    assert "analytics://models" in resource_names

    prompts = asyncio.run(server.list_prompts())
    assert "analyze_northwind" in {
        prompt.name for prompt in prompts
    }


def test_completed_mcp_payload_uses_only_sanitized_payload():
    sanitized = {
        "final_report": {
            "business_summary": "Sanitized summary",
            "key_findings": ["Sanitized finding"],
        },
        "query_result": {
            "columns": ["ProductName"],
            "rows": [{"ProductName": "Safe"}],
            "row_count": 1,
        },
    }

    state = {
        "workflow_status": "completed",
        "output_guard_result": SimpleNamespace(
            decision="allow"
        ),
        "sanitized_output_payload": sanitized,
        "final_report": SimpleNamespace(
            model_dump=lambda mode="json": {
                "business_summary": "Raw secret report"
            }
        ),
        "query_result": SimpleNamespace(
            model_dump=lambda mode="json": {
                "rows": [{"secret": "raw-value"}]
            }
        ),
        "generated_sql": "DELETE FROM Orders",
        "execution_trace": ["sensitive trace"],
    }

    payload = _build_safe_state_summary(
        state,
        include_technical_details=True,
    )

    assert payload["business_summary"] == "Sanitized summary"
    assert payload["query_result"] == sanitized["query_result"]
    assert "Raw secret report" not in repr(payload)
    assert "DELETE FROM Orders" not in repr(payload)
    assert "sensitive trace" not in repr(payload)


def test_blocked_and_review_mcp_payloads_expose_no_sensitive_state():
    forbidden = (
        "DELETE FROM Orders",
        "raw-secret",
        "system prompt",
        "C:\\database\\northwind.db",
        "trace-event",
    )

    for status in (
        "security_blocked",
        "security_review_required",
        "output_blocked",
        "output_review_required",
        "failed",
    ):
        payload = _build_safe_state_summary(
            {
                "workflow_status": status,
                "generated_sql": forbidden[0],
                "execution_trace": [forbidden[4]],
                "final_report": forbidden[1],
                "error_message": forbidden[2],
                "database_path": forbidden[3],
            },
            include_technical_details=True,
        )

        serialized = repr(payload)
        for value in forbidden:
            assert value not in serialized
