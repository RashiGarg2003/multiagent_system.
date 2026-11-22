"""
Streamlit frontend for the Multi-Agent Data Analytics System.

Architecture:

    Streamlit
        |
        | HTTP
        v
    FastAPI
        |
        v
    LangGraph
        |
        v
    Agents 1 to 5
        |
        v
    Ollama and read-only SQLite

Streamlit responsibilities:

1. Accept a business question.
2. Call the FastAPI backend.
3. Display clarification, failure, or completed results.
4. Render the chart using ChartTool.
5. Display optional technical details.

Streamlit does not directly access:

- Ollama
- LangGraph
- SQLite
- Agents
- SQL execution tools
"""

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

from models.schemas import (
    ChartSpecification,
    QueryExecutionResult,
)
from tools.chart_tool import ChartTool


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Multi-Agent Data Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------
# Frontend configuration
# ---------------------------------------------------------------------

FASTAPI_BASE_URL = os.getenv(
    "FASTAPI_BASE_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

ANALYTICS_ENDPOINT = (
    f"{FASTAPI_BASE_URL}/api/v1/analytics/query"
)

HEALTH_ENDPOINT = (
    f"{FASTAPI_BASE_URL}/health"
)

READINESS_ENDPOINT = (
    f"{FASTAPI_BASE_URL}/ready"
)

MODELS_ENDPOINT = (
    f"{FASTAPI_BASE_URL}/api/v1/models"
)

API_CONNECTION_TIMEOUT_SECONDS = 10

# Local LLM workflows can take several minutes.
API_WORKFLOW_TIMEOUT_SECONDS = 600

MAX_SQL_RETRIES = 3
MAXIMUM_CHART_ROWS = 100


# ---------------------------------------------------------------------
# Cached frontend resources
# ---------------------------------------------------------------------

@st.cache_resource(
    show_spinner=False
)
def get_chart_tool() -> ChartTool:
    """
    Create the deterministic Plotly chart renderer.

    This is a frontend visualization tool. It does not access the
    database or execute business logic.
    """

    return ChartTool(
        maximum_chart_rows=(
            MAXIMUM_CHART_ROWS
        )
    )


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------

def initialize_session_state() -> None:
    """
    Initialize values that survive Streamlit reruns.
    """

    defaults = {
        "api_response": None,
        "api_error": None,
        "last_question": "",
        "include_technical_details": True,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------
# FastAPI client functions
# ---------------------------------------------------------------------

def api_get(
    endpoint: str,
    timeout_seconds: int = (
        API_CONNECTION_TIMEOUT_SECONDS
    ),
) -> dict[str, Any]:
    """
    Send a GET request to FastAPI.
    """

    try:
        response = requests.get(
            endpoint,
            timeout=timeout_seconds,
            headers={
                "Accept": "application/json",
            },
        )

    except requests.ConnectionError as exc:
        raise ConnectionError(
            "The FastAPI backend is not available. "
            "Start it on port 8000."
        ) from exc

    except requests.Timeout as exc:
        raise TimeoutError(
            "The FastAPI request timed out."
        ) from exc

    response.raise_for_status()

    return response.json()


def check_backend_health() -> dict[str, Any]:
    """
    Check whether the FastAPI process is responding.
    """

    return api_get(
        HEALTH_ENDPOINT
    )


def check_backend_readiness() -> dict[str, Any]:
    """
    Check Ollama, database, and LangGraph readiness.
    """

    return api_get(
        READINESS_ENDPOINT
    )


def get_model_status() -> dict[str, Any]:
    """
    Retrieve required Ollama-model status through FastAPI.
    """

    return api_get(
        MODELS_ENDPOINT
    )


def submit_analytics_question(
    question: str,
    include_technical_details: bool,
) -> dict[str, Any]:
    """
    Submit a business question to FastAPI.

    Args:
        question:
            Natural-language business question.

        include_technical_details:
            Whether FastAPI should return SQL, intent, plan, and trace.

    Returns:
        Parsed API JSON response.
    """

    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError(
            "Please enter a business question."
        )

    request_body = {
        "question": cleaned_question,
        "max_sql_retries": (
            MAX_SQL_RETRIES
        ),
        "include_technical_details": (
            include_technical_details
        ),
    }

    try:
        response = requests.post(
            ANALYTICS_ENDPOINT,
            json=request_body,
            timeout=(
                API_WORKFLOW_TIMEOUT_SECONDS
            ),
            headers={
                "Accept": "application/json",
                "Content-Type": (
                    "application/json"
                ),
            },
        )

    except requests.ConnectionError as exc:
        raise ConnectionError(
            "Could not connect to the FastAPI backend. "
            "Start the backend with:\n\n"
            "python -m uvicorn api.main:app "
            "--host 127.0.0.1 --port 8000"
        ) from exc

    except requests.Timeout as exc:
        raise TimeoutError(
            "The analytics request exceeded "
            f"{API_WORKFLOW_TIMEOUT_SECONDS} seconds."
        ) from exc

    if response.status_code == 422:
        validation_payload = response.json()

        raise ValueError(
            "FastAPI rejected the request: "
            f"{validation_payload}"
        )

    if response.status_code == 503:
        raise RuntimeError(
            "The FastAPI backend is running, but "
            "Ollama, the database, or LangGraph "
            "is not ready."
        )

    if response.status_code >= 500:
        raise RuntimeError(
            "The FastAPI backend encountered an "
            f"unexpected error: {response.text}"
        )

    response.raise_for_status()

    payload = response.json()

    if not isinstance(payload, dict):
        raise RuntimeError(
            "FastAPI returned an invalid response."
        )

    return payload


# ---------------------------------------------------------------------
# API response conversion
# ---------------------------------------------------------------------

def create_query_result_model(
    api_response: dict[str, Any],
) -> QueryExecutionResult | None:
    """
    Convert the API query result into the shared Pydantic model.

    ChartTool expects QueryExecutionResult.
    """

    query_payload = api_response.get(
        "query_result"
    )

    if not isinstance(
        query_payload,
        dict,
    ):
        return None

    technical_details = (
        api_response.get(
            "technical_details"
        )
    )

    executed_sql = (
        "SELECT result returned by API"
    )

    if isinstance(
        technical_details,
        dict,
    ):
        generated_sql = (
            technical_details.get(
                "generated_sql"
            )
        )

        if (
            isinstance(generated_sql, str)
            and generated_sql.strip()
        ):
            executed_sql = (
                generated_sql.strip()
            )

    return QueryExecutionResult(
        columns=query_payload.get(
            "columns",
            [],
        ),
        rows=query_payload.get(
            "rows",
            [],
        ),
        row_count=query_payload.get(
            "row_count",
            0,
        ),
        execution_time_ms=(
            query_payload.get(
                "execution_time_ms",
                0.0,
            )
        ),
        truncated=query_payload.get(
            "truncated",
            False,
        ),
        query=executed_sql,
    )


def create_chart_specification(
    api_response: dict[str, Any],
) -> ChartSpecification | None:
    """
    Convert the API chart payload into ChartSpecification.
    """

    chart_payload = api_response.get(
        "chart"
    )

    if not isinstance(
        chart_payload,
        dict,
    ):
        return None

    return ChartSpecification.model_validate(
        chart_payload
    )


def create_result_dataframe(
    api_response: dict[str, Any],
) -> pd.DataFrame:
    """
    Convert API query-result rows into a DataFrame.
    """

    query_payload = api_response.get(
        "query_result"
    )

    if not isinstance(
        query_payload,
        dict,
    ):
        return pd.DataFrame()

    return pd.DataFrame(
        query_payload.get(
            "rows",
            [],
        ),
        columns=query_payload.get(
            "columns",
            [],
        ),
    )


# ---------------------------------------------------------------------
# Business result display
# ---------------------------------------------------------------------

def display_business_summary(
    api_response: dict[str, Any],
) -> None:
    """
    Display Agent 5's business summary.
    """

    summary = api_response.get(
        "business_summary"
    )

    if not summary:
        return

    st.subheader(
        "Business Summary"
    )

    st.write(summary)


def display_chart(
    api_response: dict[str, Any],
) -> None:
    """
    Render Agent 5's chart using ChartTool.
    """

    chart = create_chart_specification(
        api_response
    )

    query_result = (
        create_query_result_model(
            api_response
        )
    )

    if chart is None:
        return

    st.subheader(
        "Visualization"
    )

    if query_result is None:
        st.warning(
            "The API response does not contain "
            "query-result data for charting."
        )

        return

    try:
        render_result = (
            get_chart_tool().render(
                chart_specification=chart,
                query_result=query_result,
            )
        )

    except Exception as exc:
        st.warning(
            "The chart could not be rendered."
        )

        st.caption(str(exc))

        return

    if render_result.figure is not None:
        st.plotly_chart(
            render_result.figure,
            use_container_width=True,
            key="analytics_chart",
        )

    else:
        st.info(
            "No visualization is required for "
            "this result."
        )

    if render_result.warnings:
        with st.expander(
            "Chart notes",
            expanded=False,
        ):
            for warning in (
                render_result.warnings
            ):
                st.warning(warning)


def display_key_findings(
    api_response: dict[str, Any],
) -> None:
    """
    Display final key findings.
    """

    findings = api_response.get(
        "key_findings",
        [],
    )

    if not findings:
        return

    st.subheader(
        "Key Findings"
    )

    for finding in findings:
        st.markdown(
            f"- {finding}"
        )


def display_root_cause_indicators(
    api_response: dict[str, Any],
) -> None:
    """
    Display cautious root-cause indicators.
    """

    indicators = api_response.get(
        "root_cause_indicators",
        [],
    )

    if not indicators:
        return

    st.subheader(
        "Root-Cause Indicators"
    )

    st.caption(
        "These are evidence-supported indicators, "
        "not automatically proven causes."
    )

    for indicator in indicators:
        st.markdown(
            f"- {indicator}"
        )


def display_recommendations(
    api_response: dict[str, Any],
) -> None:
    """
    Display Agent 5 recommendations.
    """

    recommendations = api_response.get(
        "recommendations",
        [],
    )

    if not recommendations:
        return

    st.subheader(
        "Recommendations"
    )

    for index, recommendation in enumerate(
        recommendations,
        start=1,
    ):
        if not isinstance(
            recommendation,
            dict,
        ):
            continue

        action = recommendation.get(
            "action",
            "Recommended action",
        )

        rationale = recommendation.get(
            "rationale",
            "",
        )

        priority = recommendation.get(
            "priority",
            "medium",
        )

        with st.container(
            border=True
        ):
            st.markdown(
                f"**{index}. {action}**"
            )

            st.caption(
                f"Priority: "
                f"{str(priority).upper()}"
            )

            if rationale:
                st.write(rationale)


def display_limitations(
    api_response: dict[str, Any],
) -> None:
    """
    Display analysis and reporting limitations.
    """

    limitations = api_response.get(
        "limitations",
        [],
    )

    if not limitations:
        return

    st.subheader(
        "Limitations"
    )

    for limitation in limitations:
        st.markdown(
            f"- {limitation}"
        )


def display_query_results(
    api_response: dict[str, Any],
) -> None:
    """
    Display data returned by Agent 3b.
    """

    query_payload = api_response.get(
        "query_result"
    )

    if not isinstance(
        query_payload,
        dict,
    ):
        return

    st.subheader(
        "Data Results"
    )

    metric_columns = st.columns(4)

    metric_columns[0].metric(
        "Rows returned",
        query_payload.get(
            "row_count",
            0,
        ),
    )

    metric_columns[1].metric(
        "Columns",
        len(
            query_payload.get(
                "columns",
                [],
            )
        ),
    )

    execution_time = query_payload.get(
        "execution_time_ms",
        0.0,
    )

    metric_columns[2].metric(
        "Execution time",
        f"{execution_time:.2f} ms",
    )

    metric_columns[3].metric(
        "Result truncated",
        (
            "Yes"
            if query_payload.get(
                "truncated",
                False,
            )
            else "No"
        ),
    )

    dataframe = create_result_dataframe(
        api_response
    )

    st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------
# Technical details
# ---------------------------------------------------------------------

def display_intent_details(
    technical_details: dict[str, Any],
) -> None:
    """
    Display safe Agent 1 details.
    """

    intent = technical_details.get(
        "intent"
    )

    if intent:
        st.json(intent)

    else:
        st.info(
            "No intent details are available."
        )


def display_plan_details(
    technical_details: dict[str, Any],
) -> None:
    """
    Display safe Agent 2 details.
    """

    plan = technical_details.get(
        "plan"
    )

    if not isinstance(plan, dict):
        st.info(
            "No plan details are available."
        )

        return

    st.markdown(
        f"**Objective:** "
        f"{plan.get('objective', '')}"
    )

    st.markdown(
        f"**Strategy:** "
        f"{plan.get('strategy', '')}"
    )

    for step in plan.get(
        "steps",
        [],
    ):
        if not isinstance(step, dict):
            continue

        with st.container(
            border=True
        ):
            st.markdown(
                f"**Step "
                f"{step.get('step_id')}: "
                f"{step.get('title')}**"
            )

            st.write(
                step.get(
                    "description",
                    "",
                )
            )

            st.caption(
                f"Type: "
                f"{step.get('step_type')} | "
                f"Depends on: "
                f"{step.get('depends_on')} | "
                f"Requires SQL: "
                f"{step.get('requires_sql')}"
            )


def display_sql_details(
    technical_details: dict[str, Any],
) -> None:
    """
    Display final validated SQL.
    """

    generated_sql = (
        technical_details.get(
            "generated_sql"
        )
    )

    if generated_sql:
        st.code(
            generated_sql,
            language="sql",
        )

    else:
        st.info(
            "No generated SQL is available."
        )

    st.write(
        {
            "referenced_tables": (
                technical_details.get(
                    "referenced_tables",
                    [],
                )
            ),
            "sql_retry_count": (
                technical_details.get(
                    "sql_retry_count",
                    0,
                )
            ),
            "max_sql_retries": (
                technical_details.get(
                    "max_sql_retries",
                    0,
                )
            ),
        }
    )


def display_trace_details(
    technical_details: dict[str, Any],
) -> None:
    """
    Display the sanitized LangGraph trace.
    """

    trace = technical_details.get(
        "execution_trace",
        [],
    )

    if not trace:
        st.info(
            "No execution trace is available."
        )

        return

    st.dataframe(
        pd.DataFrame(trace),
        use_container_width=True,
        hide_index=True,
    )


def display_technical_details(
    api_response: dict[str, Any],
) -> None:
    """
    Display optional API technical details.

    RBAC will later control this section.
    """

    technical_details = (
        api_response.get(
            "technical_details"
        )
    )

    if not isinstance(
        technical_details,
        dict,
    ):
        return

    with st.expander(
        "Technical Details",
        expanded=False,
    ):
        tabs = st.tabs(
            [
                "Intent",
                "Plan",
                "Generated SQL",
                "Execution Trace",
            ]
        )

        with tabs[0]:
            display_intent_details(
                technical_details
            )

        with tabs[1]:
            display_plan_details(
                technical_details
            )

        with tabs[2]:
            display_sql_details(
                technical_details
            )

        with tabs[3]:
            display_trace_details(
                technical_details
            )


# ---------------------------------------------------------------------
# Final response display
# ---------------------------------------------------------------------

def display_completed_response(
    api_response: dict[str, Any],
) -> None:
    """
    Display a successful analytics response.
    """

    st.success(
        "Analysis completed successfully."
    )

    display_business_summary(
        api_response
    )

    display_chart(
        api_response
    )

    left_column, right_column = st.columns(
        2
    )

    with left_column:
        display_key_findings(
            api_response
        )

        display_root_cause_indicators(
            api_response
        )

    with right_column:
        display_recommendations(
            api_response
        )

        display_limitations(
            api_response
        )

    st.divider()

    display_query_results(
        api_response
    )

    display_technical_details(
        api_response
    )


def display_clarification_response(
    api_response: dict[str, Any],
) -> None:
    """
    Display a clarification-required response.
    """

    st.warning(
        "The system needs more information "
        "before continuing."
    )

    clarification_question = (
        api_response.get(
            "clarification_question"
        )
    )

    if clarification_question:
        st.markdown(
            f"### {clarification_question}"
        )

    st.info(
        "Update the business question above and "
        "submit it again."
    )

    display_technical_details(
        api_response
    )


def display_failed_response(
    api_response: dict[str, Any],
) -> None:
    """
    Display a controlled workflow failure.
    """

    st.error(
        "The analytics workflow could not "
        "complete the request."
    )

    error = api_response.get(
        "error"
    )

    if isinstance(error, dict):
        error_type = error.get(
            "error_type"
        )

        message = error.get(
            "message"
        )

        stage = error.get(
            "stage"
        )

        retryable = error.get(
            "retryable",
            False,
        )

        if error_type:
            st.markdown(
                f"**Error type:** "
                f"`{error_type}`"
            )

        if stage:
            st.markdown(
                f"**Workflow stage:** "
                f"`{stage}`"
            )

        if message:
            st.write(message)

        if retryable:
            st.info(
                "The request may succeed if submitted again."
            )

    display_technical_details(
        api_response
    )

def display_security_blocked_response(
    api_response: dict[str, Any],
) -> None:
    """
    Display a controlled security-block response.
    """

    st.error(
        "This request was blocked by "
        "application security policy."
    )

    security = api_response.get(
        "security"
    )

    if isinstance(security, dict):
        message = security.get(
            "message"
        )

        if message:
            st.write(message)

        categories = security.get(
            "risk_categories",
            [],
        )

        if categories:
            st.caption(
                "Detected risk categories: "
                + ", ".join(categories)
            )


def display_review_required_response(
    api_response: dict[str, Any],
) -> None:
    """
    Display controlled security or quality review.
    """

    # Retrieve values first
    security = api_response.get("security", {})
    output_quality = api_response.get("output_quality", {})

    # Normalize the deterministic decision safely
    deterministic_decision = ""

    if isinstance(output_quality, dict):
        deterministic_decision = str(
            output_quality.get(
                "deterministic_decision",
                "",
            )
        ).strip().lower()

    # Display the correct warning
    if deterministic_decision == "regenerate":
        st.warning(
            "The generated report failed output-quality checks "
            "and requires regeneration."
        )
    else:
        st.warning(
            "This request or generated response "
            "requires review before release."
        )

    # Display security message and categories
    if isinstance(security, dict):
        message = security.get("message")

        if message:
            st.write(message)

        categories = security.get(
            "risk_categories",
            [],
        )

        if categories:
            st.caption(
                "Detected risk categories: "
                + ", ".join(categories)
            )

    # Display output-quality information
    if isinstance(output_quality, dict):
        quality_message = output_quality.get("message")

        if quality_message:
            st.write(quality_message)

        st.caption(
            "Output validation decision: "
            f"{output_quality.get('deterministic_decision')} | "
            "LLM evaluation: "
            f"{output_quality.get('llm_evaluation_decision')}"
        )

        # Temporary debugging information
        with st.expander("Output quality details"):
            st.json(output_quality)

        
def display_api_response(
    api_response: dict[str, Any],
) -> None:
    """
    Route the FastAPI response to the correct UI.
    """

    response_status = api_response.get(
        "status"
    )

    if response_status == "completed":
        display_completed_response(
            api_response
        )

    elif (
        response_status
        == "clarification_required"
    ):
        display_clarification_response(
            api_response
        )

    elif response_status in {
        "security_blocked",
        "output_blocked",
    }:
        display_security_blocked_response(
            api_response
        )

    elif response_status in {
        "security_review_required",
        "output_review_required",
    }:
        display_review_required_response(
            api_response
        )

    elif response_status == "failed":
        display_failed_response(
            api_response
        )

    else:
        st.error(
            "FastAPI returned an unknown "
            f"workflow status: {response_status}"
        )


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

def display_sidebar() -> bool:
    """
    Display FastAPI backend status.

    Returns:
        True when the backend reports ready.
    """

    with st.sidebar:
        st.header(
            "Backend Status"
        )

        backend_ready = False

        try:
            health_payload = (
                check_backend_health()
            )

            if (
                health_payload.get(
                    "status"
                )
                == "healthy"
            ):
                st.success(
                    "FastAPI is running"
                )

            else:
                st.warning(
                    "FastAPI returned an "
                    "unexpected health status"
                )

        except Exception as exc:
            st.error(
                "FastAPI is unavailable"
            )

            st.caption(str(exc))

            st.code(
                "python -m uvicorn "
                "api.main:app --host "
                "127.0.0.1 --port 8000",
                language="powershell",
            )

            return False

        try:
            readiness_payload = (
                check_backend_readiness()
            )

            backend_ready = (
                readiness_payload.get(
                    "status"
                )
                == "ready"
            )

            st.subheader(
                "Dependencies"
            )

            for component in (
                readiness_payload.get(
                    "components",
                    [],
                )
            ):
                if not isinstance(
                    component,
                    dict,
                ):
                    continue

                name = component.get(
                    "component",
                    "component",
                )

                ready = component.get(
                    "ready",
                    False,
                )

                message = component.get(
                    "message",
                    "",
                )

                if ready:
                    st.markdown(
                        f"✅ **{name}**"
                    )

                else:
                    st.markdown(
                        f"❌ **{name}**"
                    )

                st.caption(message)

        except Exception as exc:
            st.error(
                "Backend readiness check failed"
            )

            st.caption(str(exc))

            backend_ready = False

        try:
            models_payload = (
                get_model_status()
            )

            st.subheader(
                "Ollama Models"
            )

            for model in (
                models_payload.get(
                    "models",
                    [],
                )
            ):
                if not isinstance(
                    model,
                    dict,
                ):
                    continue

                installed = model.get(
                    "installed",
                    False,
                )

                model_name = model.get(
                    "name",
                    "unknown",
                )

                icon = (
                    "✅"
                    if installed
                    else "❌"
                )

                st.markdown(
                    f"{icon} `{model_name}`"
                )

        except Exception:
            st.warning(
                "Model status is unavailable."
            )

        st.subheader(
            "Architecture"
        )

        st.markdown(
            """
1. Streamlit frontend
2. FastAPI backend
3. LangGraph orchestration
4. Five analytical agents
5. Read-only SQLite
6. Local Ollama models
            """
        )

        st.caption(
            f"API: {FASTAPI_BASE_URL}"
        )

        st.divider()

        include_details = st.checkbox(
            "Include technical details",
            value=st.session_state[
                "include_technical_details"
            ],
            help=(
                "Shows intent, plan, generated SQL, "
                "and execution trace."
            ),
        )

        st.session_state[
            "include_technical_details"
        ] = include_details

        if st.button(
            "Clear current result",
            use_container_width=True,
        ):
            st.session_state[
                "api_response"
            ] = None

            st.session_state[
                "api_error"
            ] = None

            st.session_state[
                "last_question"
            ] = ""

            st.rerun()

        return backend_ready


# ---------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------

def main() -> None:
    """
    Render the FastAPI-backed Streamlit application.
    """

    initialize_session_state()

    st.title(
        "Multi-Agent Data Analytics System"
    )

    st.write(
        "Ask a business question in natural language. "
        "Streamlit sends the request to FastAPI, which "
        "runs the five-agent LangGraph workflow against "
        "the read-only Northwind database."
    )

    backend_ready = display_sidebar()

    if not backend_ready:
        st.warning(
            "Start the FastAPI backend and confirm "
            "all dependencies are ready before "
            "submitting a question."
        )

    example_questions = [
        "Show the top 5 products by revenue.",
        "Show the number of orders by country.",
        "Compare revenue across product categories.",
        "Show the monthly revenue trend.",
        (
            "Which customers generated "
            "the highest revenue?"
        ),
    ]

    selected_example = st.selectbox(
        "Example questions",
        options=[
            "Write my own question",
            *example_questions,
        ],
    )

    if (
        selected_example
        == "Write my own question"
    ):
        default_question = (
            st.session_state.get(
                "last_question",
                "",
            )
        )

    else:
        default_question = selected_example

    with st.form(
        "business_question_form",
        clear_on_submit=False,
    ):
        user_question = st.text_area(
            "Business question",
            value=default_question,
            height=100,
            placeholder=(
                "Example: Show the top 5 "
                "products by revenue."
            ),
        )

        submitted = st.form_submit_button(
            "Analyze Data",
            type="primary",
            use_container_width=True,
            disabled=not backend_ready,
        )

    if submitted:
        cleaned_question = (
            user_question.strip()
        )

        if not cleaned_question:
            st.warning(
                "Enter a business question "
                "before starting the analysis."
            )

        else:
            st.session_state[
                "last_question"
            ] = cleaned_question

            st.session_state[
                "api_error"
            ] = None

            with st.spinner(
                "FastAPI is running the five-agent "
                "workflow. Local models may take "
                "several minutes..."
            ):
                try:
                    api_response = (
                        submit_analytics_question(
                            question=(
                                cleaned_question
                            ),
                            include_technical_details=(
                                st.session_state[
                                    "include_technical_details"
                                ]
                            ),
                        )
                    )

                    st.session_state[
                        "api_response"
                    ] = api_response

                except Exception as exc:
                    st.session_state[
                        "api_error"
                    ] = str(exc)

                    st.session_state[
                        "api_response"
                    ] = None

    api_error = st.session_state.get(
        "api_error"
    )

    if api_error:
        st.error(
            "The frontend could not complete "
            "the API request."
        )

        st.write(api_error)

    api_response = st.session_state.get(
        "api_response"
    )

    if isinstance(
        api_response,
        dict,
    ):
        st.divider()

        st.markdown(
            f"### Results for: "
            f"{st.session_state['last_question']}"
        )

        display_api_response(
            api_response
        )

    else:
        st.info(
            "Choose an example or enter a business "
            "question, then select **Analyze Data**."
        )


if __name__ == "__main__":
    main()