"""
Conditional routing functions for the LangGraph workflow.

Routing functions inspect GraphState and return the name of the next
workflow route.

These functions do not:

- Call an LLM.
- Execute database queries.
- Modify GraphState.
- Contain database-specific business logic.
- Generate user-facing reports.

Current workflow routes:

Input Guard:
    - prompt_guard
    - security_block
    - security_review
    - failure

Prompt-Injection Guard:
    - intent
    - security_block
    - security_review
    - failure

Intent Agent:
    - clarification
    - planning
    - failure

Planning Agent:
    - schema
    - failure

Schema Manager:
    - sql_generation
    - failure

SQL Generation:
    - sql_validation_execution
    - failure

SQL Validation and Execution:
    - retry_sql
    - analysis
    - failure

Analysis Agent:
    - reporting
    - failure

Reporting Agent:
    - unified_output_evaluator
    - failure

Unified Output Evaluator:
    - completed
    - output_block
    - regenerate_report
    - output_review
    - failure
"""

from typing import Literal

from workflow.state import GraphState


def enum_value(
    value: object,
) -> object:
    """
    Return a plain Enum value for routing comparisons.
    """

    if value is None:
        return None

    primitive_value = getattr(
        value,
        "value",
        value,
    )

    if isinstance(
        primitive_value,
        str,
    ):
        return primitive_value.strip().lower()

    return primitive_value


# ---------------------------------------------------------------------
# Route types
# ---------------------------------------------------------------------

InputGuardRoute = Literal[
    "prompt_guard",
    "security_block",
    "security_review",
    "failure",
]


PromptGuardRoute = Literal[
    "intent",
    "security_block",
    "security_review",
    "failure",
]


IntentRoute = Literal[
    "clarification",
    "planning",
    "failure",
]

ClarificationRoute = Literal[
    "intent_agent",
    "security_block",
    "security_review",
    "failure",
]


PlanningRoute = Literal[
    "schema",
    "failure",
]


SchemaRoute = Literal[
    "sql_generation",
    "failure",
]


SQLGenerationRoute = Literal[
    "sql_validation_execution",
    "failure",
]


SQLExecutionRoute = Literal[
    "retry_sql",
    "analysis",
    "security_block",
    "failure",
]


AnalysisRoute = Literal[
    "reporting",
    "failure",
]


ReportingRoute = Literal[
    "unified_output_evaluator",
    "failure",
]

UnifiedOutputRoute = Literal[
    "completed",
    "output_block",
    "regenerate_report",
    "output_review",
    "failure",
]
# ---------------------------------------------------------------------
# Shared failure check
# ---------------------------------------------------------------------

def has_workflow_failed(
    state: GraphState,
) -> bool:
    """
    Return True when a technical workflow failure exists.

    Controlled security and output decisions are not treated as
    technical failures.

    These controlled states include:

    - security_blocked
    - security_review_required
    - output_blocked
    - output_review_required
    """

    workflow_status = state.get(
        "workflow_status"
    )

    error_message = state.get(
        "error_message"
    )

    if workflow_status == "failed":
        return True

    return (
        isinstance(
            error_message,
            str,
        )
        and bool(
            error_message.strip()
        )
    )


# ---------------------------------------------------------------------
# Input-security routing
# ---------------------------------------------------------------------

def route_after_input_guard(
    state: GraphState,
) -> InputGuardRoute:
    """
    Decide what happens after deterministic input screening.

    Routes:

    - prompt_guard:
      Safe or safely redacted input continues to semantic screening.

    - security_block:
      A prohibited operation or restricted value was detected.

    - security_review:
      The request requires human security review.

    - failure:
      Input-guard execution failed technically or returned an
      inconsistent state.
    """

    if has_workflow_failed(state):
        return "failure"

    guard_result = state.get(
        "input_guard_result"
    )

    if guard_result is None:
        return "failure"

    decision = state.get(
        "security_decision"
    )

    workflow_status = state.get(
        "workflow_status"
    )

    if (
        decision == "block"
        or workflow_status
        == "security_blocked"
    ):
        return "security_block"

    if (
        decision == "review"
        or state.get(
            "security_review_required",
            False,
        )
        or workflow_status
        == "security_review_required"
    ):
        return "security_review"

    if decision not in {
        "allow",
        "allow_redacted",
    }:
        return "failure"

    sanitized_question = state.get(
        "user_question"
    )

    if not isinstance(
        sanitized_question,
        str,
    ):
        return "failure"

    if not sanitized_question.strip():
        return "failure"

    return "prompt_guard"


def route_after_prompt_guard(
    state: GraphState,
) -> PromptGuardRoute:
    """
    Decide what happens after semantic prompt-risk evaluation.

    Routes:

    - intent:
      Both security layers approved the request.

    - security_block:
      Semantic screening found prohibited or malicious intent.

    - security_review:
      Semantic screening found an ambiguous protected request.

    - failure:
      Prompt-risk evaluation failed technically or returned an
      inconsistent state.
    """

    if has_workflow_failed(state):
        return "failure"

    assessment = state.get(
        "prompt_risk_assessment"
    )

    if assessment is None:
        return "failure"

    decision = state.get(
        "security_decision"
    )

    workflow_status = state.get(
        "workflow_status"
    )

    if (
        decision == "block"
        or workflow_status
        == "security_blocked"
    ):
        return "security_block"

    if (
        decision == "review"
        or state.get(
            "security_review_required",
            False,
        )
        or workflow_status
        == "security_review_required"
    ):
        return "security_review"

    if decision not in {
        "allow",
        "allow_redacted",
    }:
        return "failure"

    sanitized_question = state.get(
        "user_question"
    )

    if not isinstance(
        sanitized_question,
        str,
    ):
        return "failure"

    if not sanitized_question.strip():
        return "failure"

    return "intent"


# ---------------------------------------------------------------------
# Agent 1 routing
# ---------------------------------------------------------------------

from typing import Literal


def route_after_intent(
    state: dict,
) -> Literal["clarification", "planning"]:
    """
    Route an underspecified request to clarification.
    Route a sufficiently specific request to planning.
    """

    clarification_required = state.get(
        "clarification_required",
        False,
    )

    # Normalize unexpected model/state values.
    if isinstance(clarification_required, list):
        clarification_required = any(
            bool(value)
            for value in clarification_required
        )

    elif isinstance(clarification_required, str):
        clarification_required = (
            clarification_required.strip().lower()
            in {"true", "yes", "1"}
        )

    else:
        clarification_required = bool(
            clarification_required
        )

    if clarification_required:
        return "clarification"

    return "planning"


def route_after_clarification(
    state: GraphState,
) -> ClarificationRoute:
    """Resume intent only after a screened clarification answer."""
    if has_workflow_failed(state):
        return "failure"

    workflow_status = state.get("workflow_status")
    if workflow_status == "security_blocked":
        return "security_block"
    if workflow_status == "security_review_required":
        return "security_review"
    if workflow_status == "understanding_intent":
        return "intent_agent"
    return "failure"

# ---------------------------------------------------------------------
# Agent 2 routing
# ---------------------------------------------------------------------

def route_after_planning(
    state: GraphState,
) -> PlanningRoute:
    """
    Route a valid analytical plan to schema preparation.
    """

    if has_workflow_failed(state):
        return "failure"

    plan = state.get(
        "plan"
    )

    if plan is None:
        return "failure"

    if not plan.steps:
        return "failure"

    return "schema"


# ---------------------------------------------------------------------
# Schema routing
# ---------------------------------------------------------------------

def route_after_schema(
    state: GraphState,
) -> SchemaRoute:
    """
    Route a valid non-empty database schema to Agent 3a.
    """

    if has_workflow_failed(state):
        return "failure"

    database_schema = state.get(
        "database_schema"
    )

    if not isinstance(
        database_schema,
        str,
    ):
        return "failure"

    if not database_schema.strip():
        return "failure"

    return "sql_generation"


# ---------------------------------------------------------------------
# Agent 3a routing
# ---------------------------------------------------------------------

def route_after_sql_generation(
    state: GraphState,
) -> SQLGenerationRoute:
    """
    Route generated SQL to independent validation and execution.
    """

    if has_workflow_failed(state):
        return "failure"

    sql_result = state.get(
        "sql_result"
    )

    generated_sql = state.get(
        "generated_sql"
    )

    if sql_result is None:
        return "failure"

    if not isinstance(
        generated_sql,
        str,
    ):
        return "failure"

    if not generated_sql.strip():
        return "failure"

    if sql_result.requires_clarification:
        return "failure"

    return "sql_validation_execution"


# ---------------------------------------------------------------------
# SQL retry control
# ---------------------------------------------------------------------

def can_retry_sql(
    state: GraphState,
) -> bool:
    """
    Determine whether another SQL-generation attempt is allowed.

    Retry is allowed only when:

    - Validation or execution produced an error.
    - The retry count is lower than the configured maximum.
    """

    retry_count = state.get(
        "sql_retry_count",
        0,
    )

    max_retries = state.get(
        "max_sql_retries",
        0,
    )

    if not isinstance(
        retry_count,
        int,
    ):
        return False

    if not isinstance(
        max_retries,
        int,
    ):
        return False

    validation_errors = state.get(
        "sql_validation_errors",
        [],
    )

    execution_error = state.get(
        "sql_execution_error"
    )

    has_validation_errors = bool(
        validation_errors
    )

    has_execution_error = (
        isinstance(
            execution_error,
            str,
        )
        and bool(
            execution_error.strip()
        )
    )

    return (
        (
            has_validation_errors
            or has_execution_error
        )
        and retry_count < max_retries
    )


def route_after_sql_execution(
    state: GraphState,
) -> SQLExecutionRoute:
    """
    Decide what happens after Agent 3b.

    Routes:

    - analysis:
      SQL passed validation and executed successfully.

    - retry_sql:
      SQL failed and retry attempts remain.

    - failure:
      The error is non-retryable or retries are exhausted.
    """

    if state.get("workflow_status") == "security_blocked":
        return "security_block"

    query_result = state.get(
        "query_result"
    )

    validation_errors = state.get(
        "sql_validation_errors",
        [],
    )

    execution_error = state.get(
        "sql_execution_error"
    )

    has_sql_errors = (
        bool(validation_errors)
        or (
            isinstance(
                execution_error,
                str,
            )
            and bool(
                execution_error.strip()
            )
        )
    )

    if (
        query_result is not None
        and not has_sql_errors
    ):
        return "analysis"

    if can_retry_sql(state):
        return "retry_sql"

    return "failure"


# ---------------------------------------------------------------------
# Agent 4 routing
# ---------------------------------------------------------------------

def route_after_analysis(
    state: GraphState,
) -> AnalysisRoute:
    """
    Route a validated AnalysisResult to Agent 5.
    """

    if has_workflow_failed(state):
        return "failure"

    analysis_result = state.get(
        "analysis_result"
    )

    if analysis_result is None:
        return "failure"

    return "reporting"


# ---------------------------------------------------------------------
# Agent 5 routing
# ---------------------------------------------------------------------

def route_after_reporting(
    state: GraphState,
) -> ReportingRoute:
    """
    Route Agent 5 output to the sole unified evaluator.

    Agent 5 cannot complete the workflow directly.
    """

    if has_workflow_failed(state):
        return "failure"

    final_report = state.get(
        "final_report"
    )

    if final_report is None:
        return "failure"

    workflow_status = state.get(
        "workflow_status"
    )

    if (
        workflow_status
        != "validating_output"
    ):
        return "failure"

    return "unified_output_evaluator"


def route_after_unified_output_evaluator(
    state: GraphState,
) -> UnifiedOutputRoute:
    """Route the single final evaluator's controlled decision."""
    if has_workflow_failed(state):
        return "failure"

    decision = state.get("unified_output_decision")
    workflow_status = state.get("workflow_status")

    if decision in {"allow", "allow_redacted"}:
        if (
            workflow_status == "completed"
            and state.get("sanitized_output_payload") is not None
        ):
            return "completed"
        return "failure"

    if decision == "block":
        return (
            "output_block"
            if workflow_status == "output_blocked"
            else "failure"
        )

    if decision == "regenerate":
        if (
            workflow_status == "regenerating_report"
            and state.get("report_regeneration_count", 0)
            < state.get("max_report_regenerations", 0)
        ):
            return "regenerate_report"
        return "output_review"

    if decision == "review":
        return (
            "output_review"
            if workflow_status == "output_review_required"
            else "failure"
        )

    return "failure"
