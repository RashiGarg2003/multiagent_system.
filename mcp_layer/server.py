"""Internal stdio MCP server exposing the two guarded database tools."""

import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from database.db_manager import DatabaseManager
from database.schema_manager import SchemaManager
from models.schemas import (
    AnalysisPlan,
    IntentResult,
)
from security.models import UserRole
from security.schema_policy import SchemaPolicyValidator
from tools.sql_execution_tool import SQLExecutionTool
from tools.sql_validator import SQLValidator


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_mcp_server() -> FastMCP:
    """Create the stdio server and its single shared database toolchain."""
    database_path = os.getenv(
        "DATABASE_PATH",
        str(PROJECT_ROOT / "database" / "northwind.db"),
    )
    try:
        maximum_result_rows = int(
            os.getenv("MAXIMUM_RESULT_ROWS", "1000")
        )
    except ValueError as exc:
        raise ValueError(
            "MAXIMUM_RESULT_ROWS must be an integer."
        ) from exc

    database_manager = DatabaseManager(
        database_path=database_path,
        max_rows=maximum_result_rows,
    )
    schema_manager = SchemaManager(database_manager)
    sql_validator = SQLValidator(dialect="sqlite")
    schema_policy_validator = SchemaPolicyValidator(
        database_manager=database_manager,
        sql_validator=sql_validator,
        maximum_result_rows=maximum_result_rows,
    )
    sql_execution_tool = SQLExecutionTool(
        database_manager=database_manager,
        sql_validator=sql_validator,
        schema_policy_validator=schema_policy_validator,
        role=UserRole.ANALYST,
        explain_before_execution=True,
    )
    server = FastMCP(
        "Internal Multi-Agent Analytics Database",
        instructions=(
            "Only provide approved schema metadata and execute "
            "validated read-only analytics queries."
        ),
    )

    @server.tool(
        name="get_approved_schema",
        description=(
            "Return approved SQLite schema metadata for SQL generation."
        ),
    )
    def get_approved_schema() -> dict[str, str]:
        schema_text = schema_manager.get_schema_prompt(
            include_views=False
        ).strip()
        if not schema_text:
            raise RuntimeError(
                "The approved schema is unavailable."
            )
        return {"schema": schema_text}

    @server.tool(
        name="execute_validated_read_query",
        description=(
            "Validate and execute one read-only SQLite analytics query."
        ),
    )
    def execute_validated_read_query(
        sql: str,
        intent_json: str,
        plan_json: str,
    ) -> dict[str, Any]:
        try:
            intent = IntentResult.model_validate_json(intent_json)
            plan = AnalysisPlan.model_validate_json(plan_json)
        except ValueError:
            return {
                "success": False,
                "validation_errors": [
                    "The validated intent or analytical plan is invalid."
                ],
                "execution_error": None,
                "stage": "intent_alignment",
                "retryable": False,
            }

        outcome = sql_execution_tool.validate_and_execute(
            sql,
            intent=intent,
            plan=plan,
        )
        if not outcome.success:
            return {
                "success": False,
                "validation_errors": list(
                    outcome.validation_errors
                ),
                "execution_error": outcome.execution_error,
                "stage": outcome.stage,
                "retryable": outcome.retryable,
                "security_violation": (
                    outcome.stage == "schema_policy_validation"
                    and not outcome.retryable
                ),
            }

        if outcome.query_result is None:
            return {
                "success": False,
                "validation_errors": [
                    "SQL execution returned no query result."
                ],
                "execution_error": None,
                "stage": "result_validation",
                "retryable": False,
            }

        return {
            "success": True,
            "query_result": outcome.query_result.model_dump(
                mode="json"
            ),
            "stage": outcome.stage,
            "retryable": False,
        }

    return server


def main() -> None:
    create_mcp_server().run(transport="stdio")


if __name__ == "__main__":
    main()
