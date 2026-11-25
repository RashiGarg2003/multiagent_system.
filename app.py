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

import json
import os
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from models.schemas import (
    ChartSpecification,
    QueryExecutionResult,
)
from tools.chart_tool import ChartTool

# =========================================================
# Result table and visualization helpers
# =========================================================

def format_column_name(column_name: str) -> str:
    """
    Convert database-style column names into readable labels.

    Example:
        orders_count -> Orders Count
        total_revenue -> Total Revenue
    """
    return str(column_name).replace("_", " ").strip().title()


def convert_result_to_dataframe(result_data) -> pd.DataFrame:
    """
    Safely convert different API result formats into a DataFrame.
    """

    if result_data is None:
        return pd.DataFrame()

    if isinstance(result_data, pd.DataFrame):
        return result_data.copy()

    if isinstance(result_data, list):
        return pd.DataFrame(result_data)

    if isinstance(result_data, dict):
        # Format:
        # {
        #     "columns": ["Country", "orders_count"],
        #     "rows": [["USA", 2280], ["France", 1909]]
        # }
        columns = result_data.get("columns")
        rows = result_data.get("rows")

        if columns and isinstance(rows, list):
            return pd.DataFrame(rows, columns=columns)

        # Other common API keys
        for key in [
            "data",
            "results",
            "records",
            "result",
            "query_results",
        ]:
            nested_data = result_data.get(key)

            if isinstance(nested_data, list):
                return pd.DataFrame(nested_data)

    return pd.DataFrame()


def prepare_display_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the SQL result for clean Streamlit display.
    """

    if dataframe is None or dataframe.empty:
        return pd.DataFrame()

    display_df = dataframe.copy()

    # Convert numeric-looking object columns into numeric columns.
    for column in display_df.columns:
        if display_df[column].dtype == "object":
            converted_column = pd.to_numeric(
                display_df[column],
                errors="coerce",
            )

            original_non_null = display_df[column].notna().sum()
            converted_non_null = converted_column.notna().sum()

            if (
                original_non_null > 0
                and original_non_null == converted_non_null
            ):
                display_df[column] = converted_column

    # Clean database-style column names.
    display_df.columns = [
        format_column_name(column)
        for column in display_df.columns
    ]

    return display_df


def build_result_column_config(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Create Streamlit column formatting dynamically.
    """

    column_config = {}

    for column in dataframe.columns:
        if pd.api.types.is_integer_dtype(dataframe[column]):
            column_config[column] = st.column_config.NumberColumn(
                column,
                format="%d",
                width="medium",
            )

        elif pd.api.types.is_float_dtype(dataframe[column]):
            column_lower = column.lower()

            if any(
                keyword in column_lower
                for keyword in [
                    "revenue",
                    "sales",
                    "amount",
                    "value",
                    "price",
                    "cost",
                    "total",
                ]
            ):
                column_config[column] = (
                    st.column_config.NumberColumn(
                        column,
                        format="%.2f",
                        width="medium",
                    )
                )
            else:
                column_config[column] = (
                    st.column_config.NumberColumn(
                        column,
                        format="%.2f",
                        width="medium",
                    )
                )

        else:
            column_config[column] = st.column_config.TextColumn(
                column,
                width="medium",
            )

    return column_config


def detect_date_column(
    dataframe: pd.DataFrame,
) -> str | None:
    """
    Detect a likely date/month/year column.
    """

    date_keywords = [
        "date",
        "month",
        "year",
        "quarter",
        "week",
        "day",
        "period",
    ]

    for column in dataframe.columns:
        column_lower = str(column).lower()

        if any(
            keyword in column_lower
            for keyword in date_keywords
        ):
            converted = pd.to_datetime(
                dataframe[column],
                errors="coerce",
            )

            if converted.notna().sum() > 0:
                return column

    return None


def build_fallback_chart(
    dataframe: pd.DataFrame,
):
    """
    Build a deterministic Plotly chart when the backend or
    visualization agent does not return a valid chart.

    Rules:
    1. Date + numeric column -> line chart
    2. Categorical + numeric column -> horizontal bar chart
    3. Two numeric columns -> scatter chart
    """

    if dataframe is None or dataframe.empty:
        return None

    chart_df = dataframe.copy()

    # Ensure numeric-looking values have numeric data types.
    for column in chart_df.columns:
        if chart_df[column].dtype == "object":
            converted_column = pd.to_numeric(
                chart_df[column],
                errors="coerce",
            )

            original_non_null = chart_df[column].notna().sum()
            converted_non_null = converted_column.notna().sum()

            if (
                original_non_null > 0
                and original_non_null == converted_non_null
            ):
                chart_df[column] = converted_column

    numeric_columns = (
        chart_df
        .select_dtypes(include="number")
        .columns
        .tolist()
    )

    categorical_columns = [
        column
        for column in chart_df.columns
        if column not in numeric_columns
    ]

    date_column = detect_date_column(chart_df)

    # -----------------------------------------------------
    # Date/time plus numeric value: line chart
    # -----------------------------------------------------
    if date_column and numeric_columns:
        value_column = numeric_columns[0]

        chart_df[date_column] = pd.to_datetime(
            chart_df[date_column],
            errors="coerce",
        )

        chart_df = (
            chart_df[
                [date_column, value_column]
            ]
            .dropna()
            .sort_values(date_column)
        )

        if chart_df.empty:
            return None

        figure = px.line(
            chart_df,
            x=date_column,
            y=value_column,
            markers=True,
            title=(
                f"{format_column_name(value_column)} "
                f"Trend"
            ),
            labels={
                date_column: format_column_name(date_column),
                value_column: format_column_name(value_column),
            },
        )

        figure.update_layout(
            height=460,
            margin=dict(
                l=20,
                r=20,
                t=70,
                b=20,
            ),
            hovermode="x unified",
        )

        return figure

    # -----------------------------------------------------
    # Category plus numeric value: horizontal bar chart
    # -----------------------------------------------------
    if categorical_columns and numeric_columns:
        category_column = categorical_columns[0]
        value_column = numeric_columns[0]

        chart_df = (
            chart_df[
                [category_column, value_column]
            ]
            .dropna()
            .sort_values(
                value_column,
                ascending=True,
            )
        )

        # Keep the chart readable.
        # For top-N results, retain the largest 30 values.
        if len(chart_df) > 30:
            chart_df = chart_df.tail(30)

        if chart_df.empty:
            return None

        figure = px.bar(
            chart_df,
            x=value_column,
            y=category_column,
            orientation="h",
            text=value_column,
            title=(
                f"{format_column_name(value_column)} "
                f"by {format_column_name(category_column)}"
            ),
            labels={
                category_column: format_column_name(
                    category_column
                ),
                value_column: format_column_name(
                    value_column
                ),
            },
        )

        figure.update_traces(
            texttemplate="%{text:,.0f}",
            textposition="outside",
            cliponaxis=False,
        )

        figure.update_layout(
            height=max(
                430,
                min(850, len(chart_df) * 34),
            ),
            margin=dict(
                l=20,
                r=60,
                t=70,
                b=20,
            ),
            yaxis=dict(
                categoryorder="total ascending",
            ),
            showlegend=False,
        )

        return figure

    # -----------------------------------------------------
    # Two numeric values: scatter chart
    # -----------------------------------------------------
    if len(numeric_columns) >= 2:
        x_column = numeric_columns[0]
        y_column = numeric_columns[1]

        figure = px.scatter(
            chart_df,
            x=x_column,
            y=y_column,
            title=(
                f"{format_column_name(y_column)} versus "
                f"{format_column_name(x_column)}"
            ),
            labels={
                x_column: format_column_name(x_column),
                y_column: format_column_name(y_column),
            },
        )

        figure.update_layout(
            height=460,
            margin=dict(
                l=20,
                r=20,
                t=70,
                b=20,
            ),
        )

        return figure

    return None


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
# This can be overridden by the STREAMLIT_WORKFLOW_TIMEOUT_SECONDS
# environment variable for slower machines or larger prompts.
API_WORKFLOW_TIMEOUT_SECONDS = max(
    600,
    int(
        os.getenv(
            "STREAMLIT_WORKFLOW_TIMEOUT_SECONDS",
            "1800",
        )
    ),
)

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
        "include_technical_details": False,
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
    thread_id: str | None = None,
    clarification_answer: str | None = None,
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
    if thread_id is not None and clarification_answer is not None:
        request_body["thread_id"] = thread_id
        request_body["clarification_answer"] = clarification_answer

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
        raise ValueError(
            "The request could not be validated. Check the question "
            "and any clarification answer."
        )

    if response.status_code == 503:
        raise RuntimeError(
            "The analytics service is not ready. Please try again later."
        )

    if response.status_code >= 500:
        raise RuntimeError(
            "The analytics service could not complete the request."
        )

    if response.status_code >= 400:
        raise RuntimeError(
            "The analytics request could not be completed."
        )

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


def display_key_findings(
    api_response: dict[str, Any],
) -> None:
    """
    Display the main evidence-based findings.
    """

    findings = api_response.get(
        "key_findings",
        [],
    )

    if not isinstance(
        findings,
        list,
    ) or not findings:
        return

    st.subheader(
        "Key Findings"
    )

    for finding in findings:
        if isinstance(
            finding,
            str,
        ) and finding.strip():
            st.markdown(
                f"- {finding}"
            )


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

    st.subheader("Visualization")

    if query_result is None:
        st.warning(
            "The API response does not contain "
            "query-result data for charting."
        )
        return

    visualization_df = create_result_dataframe(
        api_response
    )

    if chart is not None:
        try:
            render_result = get_chart_tool().render(
                chart_specification=chart,
                query_result=query_result,
            )

            if render_result.figure is not None:
                st.plotly_chart(
                    render_result.figure,
                    use_container_width=True,
                    key="analytics_chart",
                )

                if render_result.warnings:
                    with st.expander(
                        "Chart notes",
                        expanded=False,
                    ):
                        for warning in render_result.warnings:
                            st.warning(warning)
                return

        except Exception:
            st.caption(
                "The requested visualization could not be rendered."
            )

    if visualization_df.empty:
        st.info(
            "No visualization can be generated because "
            "the query did not return rows."
        )
        return

    numeric_columns = (
        visualization_df
        .select_dtypes(include="number")
        .columns
        .tolist()
    )
    non_numeric_columns = [
        column
        for column in visualization_df.columns
        if column not in numeric_columns
    ]
    result_is_chartable = bool(
        numeric_columns
        and (
            non_numeric_columns
            or len(numeric_columns) >= 2
        )
    )

    if not result_is_chartable:
        st.info(
            "The returned result does not contain numeric data "
            "that can be visualized."
        )
        return

    fallback_figure = build_fallback_chart(
        visualization_df
    )
    if fallback_figure is not None:
        st.plotly_chart(
            fallback_figure,
            use_container_width=True,
            config={
                "displaylogo": False,
                "responsive": True,
            },
        )
    else:
        st.info(
            "No suitable chart could be generated from "
            "the returned columns."
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

    display_df = prepare_display_dataframe(
        dataframe
    )

    if display_df.empty:
        st.info("The query did not return any data rows.")
        return

    table_header_col, table_control_col = st.columns(
        [4, 1]
    )

    with table_control_col:
        show_all_rows = st.toggle(
            "Show all",
            value=len(display_df) <= 20,
            key="show_all_query_result_rows",
        )

    if show_all_rows:
        visible_df = display_df
    else:
        visible_df = display_df.head(10)

    st.dataframe(
        visible_df,
        use_container_width=True,
        hide_index=True,
        height=min(
            600,
            38 * (len(visible_df) + 1),
        ),
        column_config=build_result_column_config(
            visible_df
        ),
    )

    csv_column, json_column = st.columns(2)
    with csv_column:
        st.download_button(
            "Download CSV",
            data=display_df.to_csv(index=False).encode("utf-8"),
            file_name="analytics-results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with json_column:
        st.download_button(
            "Download JSON",
            data=json.dumps(
                query_payload,
                ensure_ascii=True,
                indent=2,
            ),
            file_name="analytics-results.json",
            mime="application/json",
            use_container_width=True,
        )

    if not show_all_rows and len(display_df) > 10:
        st.caption(
            f"Showing 10 of {len(display_df):,} rows. "
            "Enable “Show all” to view the complete result."
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


def display_execution_trace(
    api_response: dict[str, Any],
) -> None:
    """Show the safe workflow trace in a collapsed expander."""
    trace = api_response.get("execution_trace", [])
    if not isinstance(trace, list) or not trace:
        return

    with st.expander("Execution trace", expanded=False):
        display_df = pd.DataFrame(trace)
        columns = [
            "stage",
            "status",
            "message",
            "duration_ms",
            "mcp_tool",
            "sql_retry_count",
            "report_retry_count",
        ]
        visible_columns = [
            column
            for column in columns
            if column in display_df.columns
        ]
        st.dataframe(
            display_df[visible_columns],
            use_container_width=True,
            hide_index=True,
        )


def display_quality_info(
    api_response: dict[str, Any],
) -> None:
    """Display only controlled quality scores and check outcomes."""
    quality = api_response.get("output_quality")
    if not isinstance(quality, dict):
        return

    safe_fields = {
        key: quality.get(key)
        for key in (
            "unified_decision",
            "deterministic_decision",
            "llm_evaluation_decision",
            "output_guard_decision",
            "deterministic_valid",
            "numerical_claims_valid",
            "ranking_valid",
            "chart_valid",
            "redaction_count",
            "relevance_score",
            "groundedness_score",
            "accuracy_score",
            "overall_score",
        )
        if quality.get(key) is not None
    }
    if not safe_fields:
        return

    with st.expander("Quality information", expanded=False):
        st.json(safe_fields)


def display_summary_cards(
    api_response: dict[str, Any],
) -> None:
    """Display compact executive summary indicators."""
    query_result = api_response.get("query_result")
    findings = api_response.get("key_findings", [])
    recommendations = api_response.get("recommendations", [])
    quality = api_response.get("output_quality")
    row_count = (
        query_result.get("row_count", 0)
        if isinstance(query_result, dict)
        else 0
    )
    quality_status = (
        quality.get(
            "unified_decision",
            quality.get("output_guard_decision", "Not available"),
        )
        if isinstance(quality, dict)
        else "Not available"
    )

    columns = st.columns(4)
    columns[0].metric("Rows analyzed", f"{row_count:,}")
    columns[1].metric(
        "Key findings",
        len(findings) if isinstance(findings, list) else 0,
    )
    columns[2].metric(
        "Recommendations",
        len(recommendations)
        if isinstance(recommendations, list)
        else 0,
    )
    columns[3].metric("Output status", str(quality_status).replace("_", " ").title())


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

    display_summary_cards(api_response)
    display_quality_info(api_response)

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

    original_question = api_response.get(
        "question",
        st.session_state.get(
            "last_question",
            "",
        ),
    )
    thread_id = api_response.get("thread_id")

    with st.form(
        "clarification_answer_form",
        clear_on_submit=True,
    ):
        clarification_answer = st.text_area(
            "Provide the missing information",
            height=100,
            placeholder=(
                "Enter the metric, dimensions, filters, "
                "or time period needed to continue."
            ),
        )

        answer_submitted = st.form_submit_button(
            "Continue analysis",
            type="primary",
            use_container_width=True,
        )

    if answer_submitted:
        cleaned_answer = clarification_answer.strip()

        if not cleaned_answer or not isinstance(thread_id, str):
            st.warning(
                "The clarification session is unavailable. Submit a new "
                "question to restart the analysis."
            )
        else:
            with st.status(
                "Resuming the same analysis session...",
                expanded=True,
            ) as progress:
                try:
                    revised_response = (
                        submit_analytics_question(
                            question=str(original_question),
                            include_technical_details=(
                                st.session_state[
                                    "include_technical_details"
                                ]
                            ),
                            thread_id=thread_id,
                            clarification_answer=cleaned_answer,
                        )
                    )

                    st.session_state[
                        "api_response"
                    ] = revised_response

                    st.session_state[
                        "api_error"
                    ] = None
                    progress.update(
                        label="Analysis session resumed",
                        state="complete",
                        expanded=False,
                    )

                    st.rerun()

                except Exception:
                    st.session_state[
                        "api_error"
                    ] = (
                        "The clarification could not be processed. "
                        "Please try again."
                    )
                    st.session_state[
                        "api_response"
                    ] = None

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

    st.info(
        "No results were released. You can try the question again."
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
                "unified_decision",
                output_quality.get("deterministic_decision", ""),
            )
        ).strip().lower()

    # Display the correct warning
    if deterministic_decision == "regenerate":
        st.warning(
            "The generated report failed output-quality checks "
            "and requires regeneration or review."
        )
    else:
        st.warning(
            "This request or generated response "
            "requires review before release."
        )

    # Display security message and categories
    security_message = None

    if isinstance(security, dict):
        security_message = security.get("message")

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
        if (
            quality_message
            and quality_message != security_message
        ):
            st.write(quality_message)

        st.caption(
            "Unified output decision: "
            f"{output_quality.get('unified_decision', 'Not available')}"
        )

        
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

        except Exception:
            st.error(
                "FastAPI is unavailable"
            )

            st.caption("Check that the local backend is running.")

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

        except Exception:
            st.error(
                "Backend readiness check failed"
            )

            st.caption("Backend dependency status is unavailable.")

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

    with st.form(
        "business_question_form",
        clear_on_submit=False,
    ):
        user_question = st.text_area(
            "Business question",
            value=st.session_state.get(
                "last_question",
                "",
            ),
            height=100,
            placeholder="What would you like to analyze?",
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

            with st.status(
                "Running analytics workflow...",
                expanded=True,
            ) as progress:
                try:
                    st.write(
                        "Input security → intent and plan → validated SQL "
                        "→ analysis and report → unified quality checks"
                    )
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
                    progress.update(
                        label="Workflow response received",
                        state="complete",
                        expanded=False,
                    )

                except Exception:
                    st.session_state[
                        "api_error"
                    ] = (
                        "The analytics service could not complete the "
                        "request. Please try again."
                    )

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

        st.caption(api_error)

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
        display_execution_trace(api_response)

    else:
        st.info(
            "Enter a business question to begin your analysis."
        )


if __name__ == "__main__":
    main()