"""
Model Context Protocol server for the Multi-Agent Analytics project.

The MCP server exposes:

- Read-only database schema and SQL tools.
- The complete LangGraph analytics workflow.
- Backend health/model status.
- Schema resources and a reusable analytics prompt.

Run over stdio:

    python mcp_server.py

Run through FastAPI:

    http://127.0.0.1:8000/mcp
"""

import asyncio
from typing import Any

from mcp.server.fastmcp import FastMCP

from api.dependencies import (
    AppServices,
    get_services,
)
from database.db_manager import DatabaseManager
from models.schemas import (
    IntentResult,
)
from security.models import UserRole
from tools.sql_execution_tool import SQLExecutionTool
from tools.sql_validator import SQLValidator
from workflow.state import (
    create_initial_state,
)


WORKFLOW_SEMAPHORE = asyncio.Semaphore(1)


def _service_container() -> AppServices:
    return get_services()


def _enum_value(value: Any) -> Any:
    if value is None:
        return None

    primitive_value = getattr(value, "value", value)

    if isinstance(primitive_value, str):
        return primitive_value.strip().lower()

    return primitive_value


def _build_database_schema(
    database_manager: DatabaseManager,
) -> dict[str, Any]:
    tables: dict[str, list[dict[str, Any]]] = {}

    for table_name in database_manager.get_tables():
        tables[table_name] = (
            database_manager.get_columns(table_name)
        )

    return {
        "database": str(
            database_manager.database_path
        ),
        "tables": tables,
        "views": database_manager.get_views(),
    }


def _build_safe_state_summary(
    state: dict[str, Any],
    *,
    include_technical_details: bool,
) -> dict[str, Any]:
    status = state.get("workflow_status")
    final_report = state.get("final_report")
    query_result = state.get("query_result")
    output_quality = state.get(
        "deterministic_validation_result"
    )
    error_type = state.get("error_type")
    error_message = state.get("error_message")

    report_payload: dict[str, Any] = {}

    if final_report is not None:
        report_payload = (
            final_report.model_dump(mode="json")
        )

    query_payload: dict[str, Any] | None = None

    if query_result is not None:
        query_payload = (
            query_result.model_dump(mode="json")
        )

    payload: dict[str, Any] = {
        "status": status,
        "question": state.get("user_question"),
        "question_was_redacted": bool(
            state.get("question_was_redacted", False)
        ),
        "security_decision": state.get(
            "security_decision"
        ),
        "clarification_required": bool(
            state.get("clarification_required", False)
        ),
        "clarification_question": state.get(
            "clarification_question"
        ),
        "business_summary": report_payload.get(
            "business_summary"
        ),
        "key_findings": report_payload.get(
            "key_findings",
            [],
        ),
        "root_cause_indicators": report_payload.get(
            "root_cause_indicators",
            [],
        ),
        "recommendations": report_payload.get(
            "recommendations",
            [],
        ),
        "chart": report_payload.get("chart"),
        "limitations": report_payload.get(
            "limitations",
            [],
        ),
        "query_result": query_payload,
        "output_quality": (
            output_quality.model_dump(mode="json")
            if hasattr(output_quality, "model_dump")
            else None
        ),
        "error": (
            {
                "error_type": error_type,
                "message": error_message,
                "stage": state.get("current_node"),
            }
            if status == "failed"
            else None
        ),
    }

    if include_technical_details:
        payload["technical_details"] = {
            "intent": (
                state.get("intent").model_dump(mode="json")
                if isinstance(
                    state.get("intent"),
                    IntentResult,
                )
                else None
            ),
            "plan": (
                state.get("plan").model_dump(mode="json")
                if hasattr(state.get("plan"), "model_dump")
                else None
            ),
            "generated_sql": state.get("generated_sql"),
            "sql_retry_count": state.get(
                "sql_retry_count",
                0,
            ),
            "execution_trace": state.get(
                "execution_trace",
                [],
            ),
        }

    return payload


def create_mcp_server() -> FastMCP:
    server = FastMCP(
        "Multi-Agent Analytics",
        instructions=(
            "Use MCP tools and resources to inspect the read-only "
            "Northwind database and run the complete guarded analytics "
            "workflow."
        ),
        host="127.0.0.1",
        port=8001,
        mount_path="/",
        streamable_http_path="/",
        stateless_http=True,
    )

    @server.resource(
        "northwind://schema",
        name="Northwind database schema",
        description=(
            "Tables, views, and column metadata for the read-only "
            "Northwind SQLite database."
        ),
        mime_type="application/json",
    )
    async def northwind_schema() -> dict[str, Any]:
        services = _service_container()
        return _build_database_schema(
            services.database_manager
        )

    @server.resource(
        "analytics://models",
        name="Ollama model status",
        description="Availability of required local Ollama models.",
        mime_type="application/json",
    )
    async def analytics_models() -> dict[str, Any]:
        services = _service_container()

        ollama_available = (
            services.ollama_manager
            .check_server_available()
        )

        models = [
            {
                "name": model_name,
                "installed": (
                    services.ollama_manager
                    .check_model_installed(model_name)
                    if ollama_available
                    else False
                ),
            }
            for model_name
            in services.settings.required_models
        ]

        return {
            "ollama_available": ollama_available,
            "models": models,
        }

    @server.tool(
        name="list_database_objects",
        description=(
            "List tables and views in the read-only Northwind database."
        ),
    )
    async def list_database_objects() -> dict[str, Any]:
        services = _service_container()

        return {
            "tables": (
                services.database_manager.get_tables()
            ),
            "views": (
                services.database_manager.get_views()
            ),
        }

    @server.tool(
        name="get_table_columns",
        description=(
            "Get column metadata for one Northwind table or view."
        ),
    )
    async def get_table_columns(
        object_name: str,
    ) -> dict[str, Any]:
        services = _service_container()

        cleaned_name = object_name.strip()
        if not cleaned_name:
            raise ValueError(
                "object_name cannot be empty."
            )

        available = set(
            services.database_manager.get_tables()
        )
        available.update(
            services.database_manager.get_views()
        )

        if cleaned_name not in available:
            raise ValueError(
                f"Unknown table or view: {cleaned_name}"
            )

        return {
            "object_name": cleaned_name,
            "columns": (
                services.database_manager
                .get_columns(cleaned_name)
            ),
        }

    @server.tool(
        name="execute_read_sql",
        description=(
            "Validate and execute one read-only SQL query against "
            "the Northwind database."
        ),
    )
    async def execute_read_sql(
        sql: str,
        role: str = "analyst",
    ) -> dict[str, Any]:
        services = _service_container()

        try:
            normalized_role = UserRole(role.strip().lower())
        except ValueError as exc:
            raise ValueError(
                "role must be one of: viewer, analyst, admin."
            ) from exc

        sql_tool = SQLExecutionTool(
            database_manager=(
                services.database_manager
            ),
            sql_validator=SQLValidator(
                dialect="sqlite"
            ),
            role=normalized_role,
            explain_before_execution=True,
        )

        outcome = await asyncio.to_thread(
            sql_tool.validate_and_execute,
            sql,
        )

        if not outcome.success:
            return {
                "success": False,
                "validated_sql": (
                    outcome.validated_sql
                ),
                "validation_errors": (
                    outcome.validation_errors
                ),
                "execution_error": (
                    outcome.execution_error
                ),
                "stage": outcome.stage,
                "retryable": outcome.retryable,
            }

        if outcome.query_result is None:
            return {
                "success": False,
                "validation_errors": [
                    "SQL execution returned no query result."
                ],
                "stage": "result_validation",
                "retryable": False,
            }

        return {
            "success": True,
            "validated_sql": outcome.validated_sql,
            "query_result": (
                outcome.query_result
                .model_dump(mode="json")
            ),
        }

    @server.tool(
        name="run_analytics",
        description=(
            "Run the complete guarded LangGraph analytics workflow "
            "for a natural-language business question."
        ),
    )
    async def run_analytics(
        question: str,
        max_sql_retries: int = 3,
        include_technical_details: bool = False,
    ) -> dict[str, Any]:
        services = _service_container()

        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError(
                "question cannot be empty."
            )

        if not 3 <= len(cleaned_question) <= 2000:
            raise ValueError(
                "question must contain 3 to 2000 characters."
            )

        if not 0 <= max_sql_retries <= 5:
            raise ValueError(
                "max_sql_retries must be between 0 and 5."
            )

        effective_retries = min(
            max_sql_retries,
            services.settings.maximum_sql_retries,
        )

        initial_state = create_initial_state(
            user_question=cleaned_question,
            max_sql_retries=effective_retries,
        )

        async with WORKFLOW_SEMAPHORE:
            final_state = await asyncio.to_thread(
                services.graph.invoke,
                initial_state,
                config={
                    "recursion_limit": (
                        services.settings
                        .langgraph_recursion_limit
                    )
                },
            )

        return _build_safe_state_summary(
            final_state,
            include_technical_details=(
                include_technical_details
                and services.settings
                .expose_technical_details
            ),
        )

    @server.prompt(
        name="analyze_northwind",
        description=(
            "Build an evidence-grounded Northwind analytics request."
        ),
    )
    async def analyze_northwind_prompt(
        question: str,
    ) -> str:
        return (
            "Analyze the following Northwind business question "
            "using the guarded analytics workflow.\n\n"
            f"Question: {question}\n\n"
            "Use the database schema resource when needed. "
            "Execute only read-only SQL. Report limitations and "
            "do not invent unsupported numbers or causes."
        )

    return server


def main() -> None:
    create_mcp_server().run(
        transport="stdio"
    )


if __name__ == "__main__":
    main()
