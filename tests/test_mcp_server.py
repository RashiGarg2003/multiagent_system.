"""
Focused tests for MCP server discovery and safe database tools.
"""

import asyncio

from mcp_server import create_mcp_server


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
