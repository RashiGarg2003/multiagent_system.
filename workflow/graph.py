"""
LangGraph orchestration for the Multi-Agent Data Analytics System.

Complete workflow:

    START
      |
      v
    Input Guard
      |
      +---- blocked --------------------> Security Block ----> END
      |
      +---- review required ------------> Security Review ---> END
      |
      v
    Prompt-Injection Guard
      |
      +---- blocked --------------------> Security Block ----> END
      |
      +---- review required ------------> Security Review ---> END
      |
      v
    Agent 1: Intent Understanding
      |
      +---- clarification required -----> Clarification ------> END
      |
      +---- failure --------------------> Failure ------------> END
      |
      v
    Agent 2: Planning
      |
      v
    Schema Manager
      |
      v
    Agent 3a: SQL Generation
      |
      v
    Agent 3b: SQL Validation and Execution
      |
      +---- retryable SQL error --------> Agent 3a
      |
      +---- retry limit reached --------> Failure ------------> END
      |
      v
    Agent 4: Analysis
      |
      v
    Agent 5: Reporting
      |
      v
    Unified Output Evaluator
      |
      +---- regenerate (max 2) ---------> Agent 5
      +---- blocked/review -------------> END
      +---- blocked --------------------> END
      |
      +---- review required ------------> END
      |
      v
    COMPLETED
      |
      v
    END

All database access remains read-only.

Output regeneration is bounded by the configured retry limit. A regenerate
decision retries report generation when capacity remains; otherwise it
produces output_review_required and safely ends the workflow.
"""

from time import perf_counter
from typing import Any, Callable
from uuid import uuid4

from langgraph.graph import (
    END,
    START,
    StateGraph,
)
from langgraph.checkpoint.memory import MemorySaver

from workflow.nodes import (
    WorkflowNodes,
    build_workflow_nodes,
)

from workflow.routing import (
    route_after_analysis,
    route_after_clarification,
    route_after_input_guard,
    route_after_intent,
    route_after_planning,
    route_after_prompt_guard,
    route_after_reporting,
    route_after_schema,
    route_after_sql_execution,
    route_after_sql_generation,
    route_after_unified_output_evaluator,
)

from workflow.state import (
    GraphState,
    create_initial_state,
)
from mcp_layer.client import InternalMCPClient


def _with_safe_trace_timing(
    node_name: str,
    node_callable: Callable[[GraphState], dict[str, Any]],
) -> Callable[[GraphState], dict[str, Any]]:
    """Measure a node and enrich only its final safe trace event."""

    def invoke(state: GraphState) -> dict[str, Any]:
        started = perf_counter()
        update = node_callable(state)
        if not isinstance(update, dict):
            raise TypeError(
                f"Workflow node '{node_name}' returned an invalid update."
            )

        trace = list(
            update.get(
                "execution_trace",
                state.get("execution_trace", []),
            )
        )
        if trace and trace[-1].get("node") == node_name:
            event = dict(trace[-1])
            event["stage"] = node_name
            event["duration_ms"] = round(
                (perf_counter() - started) * 1000,
                2,
            )
            event["sql_retry_count"] = update.get(
                "sql_retry_count",
                state.get("sql_retry_count", 0),
            )
            event["report_retry_count"] = update.get(
                "report_regeneration_count",
                state.get("report_regeneration_count", 0),
            )
            trace[-1] = event
            update["execution_trace"] = trace
        return update

    return invoke


# ---------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------

def build_graph(
    workflow_nodes: WorkflowNodes,
    checkpointer: Any = None,
) -> Any:
    """
    Build and compile the complete LangGraph workflow.

    Args:
        workflow_nodes:
            Dependency-injected collection containing:

            - Input Guard
            - Prompt-Injection Guard
            - Agents 1 through 5
            - Database schema preparation
            - SQL validation and execution
            - Unified post-Agent-5 output evaluation
            - Clarification and controlled failure nodes

    Returns:
        Compiled LangGraph application.
    """

    if not isinstance(
        workflow_nodes,
        WorkflowNodes,
    ):
        raise TypeError(
            "workflow_nodes must be a "
            "WorkflowNodes instance."
        )

    graph_builder = StateGraph(
        GraphState
    )

    def add_timed_node(
        node_name: str,
        node_callable: Callable[[GraphState], dict[str, Any]],
    ) -> None:
        graph_builder.add_node(
            node_name,
            _with_safe_trace_timing(
                node_name,
                node_callable,
            ),
        )

    # -----------------------------------------------------------------
    # Register input-security nodes
    # -----------------------------------------------------------------

    add_timed_node(
        "input_guard",
        workflow_nodes.input_guard_node,
    )

    add_timed_node(
        "prompt_injection_guard",
        (
            workflow_nodes
            .prompt_injection_guard_node
        ),
    )

    add_timed_node(
        "security_block",
        workflow_nodes.security_block_node,
    )

    add_timed_node(
        "security_review",
        workflow_nodes.security_review_node,
    )

    # -----------------------------------------------------------------
    # Register analytical workflow nodes
    # -----------------------------------------------------------------

    add_timed_node(
        "intent_agent",
        workflow_nodes.intent_node,
    )

    add_timed_node(
        "clarification",
        workflow_nodes.clarification_node,
    )

    add_timed_node(
        "planning_agent",
        workflow_nodes.planning_node,
    )

    add_timed_node(
        "schema_manager",
        workflow_nodes.schema_node,
    )

    add_timed_node(
        "sql_generation_agent",
        workflow_nodes.sql_generation_node,
    )

    add_timed_node(
        "sql_validation_execution",
        (
            workflow_nodes
            .sql_validation_execution_node
        ),
    )

    add_timed_node(
        "analysis_agent",
        workflow_nodes.analysis_node,
    )

    add_timed_node(
        "reporting_agent",
        workflow_nodes.reporting_node,
    )

    # -----------------------------------------------------------------
    # Register output-validation and protection nodes
    # -----------------------------------------------------------------

    add_timed_node(
        "unified_output_evaluator",
        workflow_nodes.unified_output_evaluator_node,
    )

    # -----------------------------------------------------------------
    # Register technical failure node
    # -----------------------------------------------------------------

    add_timed_node(
        "failure",
        workflow_nodes.failure_node,
    )

    # -----------------------------------------------------------------
    # Start workflow
    # -----------------------------------------------------------------

    graph_builder.add_edge(
        START,
        "input_guard",
    )

    # -----------------------------------------------------------------
    # Deterministic input-security routing
    # -----------------------------------------------------------------

    graph_builder.add_conditional_edges(
        "input_guard",
        route_after_input_guard,
        {
            "prompt_guard": (
                "prompt_injection_guard"
            ),
            "security_block": (
                "security_block"
            ),
            "security_review": (
                "security_review"
            ),
            "failure": "failure",
        },
    )

    # -----------------------------------------------------------------
    # Semantic prompt-injection routing
    # -----------------------------------------------------------------

    graph_builder.add_conditional_edges(
        "prompt_injection_guard",
        route_after_prompt_guard,
        {
            "intent": "intent_agent",
            "security_block": (
                "security_block"
            ),
            "security_review": (
                "security_review"
            ),
            "failure": "failure",
        },
    )

    # -----------------------------------------------------------------
    # Controlled input-security termination
    # -----------------------------------------------------------------

    graph_builder.add_edge(
        "security_block",
        END,
    )

    graph_builder.add_edge(
        "security_review",
        END,
    )

    # -----------------------------------------------------------------
    # Agent 1 routing
    # -----------------------------------------------------------------

    graph_builder.add_conditional_edges(
        "intent_agent",
        route_after_intent,
        {
            "clarification": (
                "clarification"
            ),
            "planning": (
                "planning_agent"
            ),
            "failure": "failure",
        },
    )

    graph_builder.add_conditional_edges(
        "clarification",
        route_after_clarification,
        {
            "intent_agent": "intent_agent",
            "security_block": "security_block",
            "security_review": "security_review",
            "failure": "failure",
            "clarification_required": END,
        },
    )

    # -----------------------------------------------------------------
    # Agent 2 routing
    # -----------------------------------------------------------------

    graph_builder.add_conditional_edges(
        "planning_agent",
        route_after_planning,
        {
            "schema": "schema_manager",
            "failure": "failure",
        },
    )

    # -----------------------------------------------------------------
    # Database-schema routing
    # -----------------------------------------------------------------

    graph_builder.add_conditional_edges(
        "schema_manager",
        route_after_schema,
        {
            "sql_generation": (
                "sql_generation_agent"
            ),
            "failure": "failure",
        },
    )

    # -----------------------------------------------------------------
    # Agent 3a routing
    # -----------------------------------------------------------------

    graph_builder.add_conditional_edges(
        "sql_generation_agent",
        route_after_sql_generation,
        {
            "sql_validation_execution": (
                "sql_validation_execution"
            ),
            "failure": "failure",
        },
    )

    # -----------------------------------------------------------------
    # Agent 3b routing and SQL retry loop
    # -----------------------------------------------------------------

    graph_builder.add_conditional_edges(
        "sql_validation_execution",
        route_after_sql_execution,
        {
            # Retryable validation, schema-policy, compilation,
            # or execution error.
            "retry_sql": (
                "sql_generation_agent"
            ),

            # Validated query result continues to Agent 4.
            "analysis": "analysis_agent",
            "security_block": "security_block",

            # Retry limit reached or non-retryable failure.
            "failure": "failure",
        },
    )

    # -----------------------------------------------------------------
    # Agent 4 routing
    # -----------------------------------------------------------------

    graph_builder.add_conditional_edges(
        "analysis_agent",
        route_after_analysis,
        {
            "reporting": (
                "reporting_agent"
            ),
            "failure": "failure",
        },
    )

    # -----------------------------------------------------------------
    # Agent 5 routing
    # -----------------------------------------------------------------

    graph_builder.add_conditional_edges(
        "reporting_agent",
        route_after_reporting,
        {
            "unified_output_evaluator": (
                "unified_output_evaluator"
            ),
            "failure": "failure",
        },
    )

    graph_builder.add_conditional_edges(
        "unified_output_evaluator",
        route_after_unified_output_evaluator,
        {
            "completed": END,
            "output_block": END,
            "output_review": END,
            "regenerate_report": "reporting_agent",
            "failure": "failure",
        },
    )

    # -----------------------------------------------------------------
    # Controlled technical failure termination
    # -----------------------------------------------------------------

    graph_builder.add_edge(
        "failure",
        END,
    )

    return graph_builder.compile(
        checkpointer=checkpointer
    )


# ---------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------

def create_graph(
    mcp_client: InternalMCPClient,
    checkpointer: Any = None,
) -> Any:
    """
    Create the compiled workflow using configured project components.

    FastAPI initializes and reuses this compiled graph.
    """

    workflow_nodes = (
        build_workflow_nodes(mcp_client)
    )

    return build_graph(
        workflow_nodes=workflow_nodes,
        checkpointer=(
            checkpointer
            if checkpointer is not None
            else MemorySaver()
        ),
    )


# ---------------------------------------------------------------------
# Workflow execution
# ---------------------------------------------------------------------

def run_workflow(
    user_question: str,
    max_sql_retries: int = 3,
    max_report_regenerations: int = 2,
    recursion_limit: int = 50,
) -> GraphState:
    """
    Execute the complete LangGraph workflow.

    Args:
        user_question:
            Natural-language analytical question.

        max_sql_retries:
            Maximum number of failed SQL correction attempts.

        max_report_regenerations:
            Maximum number of Agent 5 report-regeneration attempts.

        recursion_limit:
            Maximum number of LangGraph execution steps.

    Returns:
        Final GraphState.
    """

    if not isinstance(
        recursion_limit,
        int,
    ):
        raise TypeError(
            "recursion_limit must be an integer."
        )

    if recursion_limit < 1:
        raise ValueError(
            "recursion_limit must be greater than zero."
        )

    thread_id = str(uuid4())
    initial_state = create_initial_state(
        user_question=user_question,
        max_sql_retries=(
            max_sql_retries
        ),
        max_report_regenerations=(
            max_report_regenerations
        ),
    )
    initial_state["thread_id"] = thread_id

    with InternalMCPClient() as mcp_client:
        graph = create_graph(mcp_client)
        final_state = graph.invoke(
            initial_state,
            config={
                "configurable": {
                    "thread_id": thread_id,
                },
                "recursion_limit": (
                    recursion_limit
                ),
            },
        )

    return final_state


# ---------------------------------------------------------------------
# Safe terminal output
# ---------------------------------------------------------------------

def print_workflow_result(
    final_state: GraphState,
) -> None:
    """
    Print a safe summary of one workflow execution.

    The original question, full schema, prompts, secrets, environment
    values, and unrestricted result data are intentionally not printed.
    """

    print("\nWorkflow result:")

    print(
        f"Status                       : "
        f"{final_state.get('workflow_status')}"
    )

    print(
        f"Current node                 : "
        f"{final_state.get('current_node')}"
    )

    print(
        f"Security decision            : "
        f"{final_state.get('security_decision')}"
    )

    print(
        f"Question redacted            : "
        f"{final_state.get('question_was_redacted')}"
    )

    print(
        f"Security review required     : "
        f"{final_state.get('security_review_required')}"
    )

    security_message = final_state.get(
        "security_message"
    )

    if security_message:
        print(
            f"Security message             : "
            f"{security_message}"
        )

    print(
        f"Clarification required       : "
        f"{final_state.get('clarification_required')}"
    )

    print(
        f"SQL retries                  : "
        f"{final_state.get('sql_retry_count', 0)}/"
        f"{final_state.get('max_sql_retries', 0)}"
    )

    print(
        f"Report regenerations         : "
        f"{final_state.get('report_regeneration_count', 0)}/"
        f"{final_state.get('max_report_regenerations', 0)}"
    )

    clarification_question = (
        final_state.get(
            "clarification_question"
        )
    )

    if clarification_question:
        print(
            f"Clarification question       : "
            f"{clarification_question}"
        )

    intent = final_state.get(
        "intent"
    )

    if intent is not None:
        print(
            f"Analysis type                 : "
            f"{intent.analysis_type}"
        )

        print(
            f"Metrics                       : "
            f"{intent.metrics}"
        )

        print(
            f"Dimensions                    : "
            f"{intent.dimensions}"
        )

    plan = final_state.get(
        "plan"
    )

    if plan is not None:
        print(
            f"Plan objective                : "
            f"{plan.objective}"
        )

        print(
            f"Plan steps                    : "
            f"{len(plan.steps)}"
        )

    database_schema = final_state.get(
        "database_schema"
    )

    print(
        f"Schema prepared               : "
        f"{bool(database_schema)}"
    )

    generated_sql = final_state.get(
        "generated_sql"
    )

    if generated_sql:
        print("\nGenerated SQL:")
        print(generated_sql)

    validation_errors = final_state.get(
        "sql_validation_errors",
        [],
    )

    if validation_errors:
        print("\nSQL validation errors:")

        for error in validation_errors:
            print(f"- {error}")

    execution_error = final_state.get(
        "sql_execution_error"
    )

    if execution_error:
        print("\nSQL execution error:")
        print(execution_error)

    query_result = final_state.get(
        "query_result"
    )

    if query_result is not None:
        print("\nQuery result:")

        print(
            f"Columns                       : "
            f"{query_result.columns}"
        )

        print(
            f"Rows returned                 : "
            f"{query_result.row_count}"
        )

        print(
            f"Truncated                     : "
            f"{query_result.truncated}"
        )

        print(
            f"Execution time                : "
            f"{query_result.execution_time_ms} ms"
        )

        preview_size = min(
            10,
            len(query_result.rows),
        )

        print(
            f"\nFirst {preview_size} rows:"
        )

        for row in query_result.rows[
            :preview_size
        ]:
            print(row)

    analysis_result = final_state.get(
        "analysis_result"
    )

    if analysis_result is not None:
        print("\nAnalysis result:")

        print(
            f"Summary: "
            f"{analysis_result.summary}"
        )

        if analysis_result.findings:
            print("\nAnalysis findings:")

            for finding in (
                analysis_result.findings
            ):
                print(
                    f"- {finding.title}: "
                    f"{finding.description}"
                )

                for evidence in (
                    finding.evidence
                ):
                    print(
                        f"  Evidence: {evidence}"
                    )

        if analysis_result.trends:
            print("\nTrends:")

            for trend in (
                analysis_result.trends
            ):
                print(f"- {trend}")

        if analysis_result.anomalies:
            print("\nAnomalies:")

            for anomaly in (
                analysis_result.anomalies
            ):
                print(f"- {anomaly}")

        if (
            analysis_result
            .possible_drivers
        ):
            print("\nPossible drivers:")

            for driver in (
                analysis_result
                .possible_drivers
            ):
                print(f"- {driver}")

        if analysis_result.limitations:
            print("\nAnalysis limitations:")

            for limitation in (
                analysis_result.limitations
            ):
                print(f"- {limitation}")

    final_report = final_state.get(
        "final_report"
    )

    if final_report is not None:
        print("\nFinal business report:")

        print(
            f"Business summary: "
            f"{final_report.business_summary}"
        )

        if final_report.key_findings:
            print("\nReport findings:")

            for finding in (
                final_report.key_findings
            ):
                print(f"- {finding}")

        if (
            final_report
            .root_cause_indicators
        ):
            print(
                "\nRoot-cause indicators:"
            )

            for indicator in (
                final_report
                .root_cause_indicators
            ):
                print(f"- {indicator}")

        if final_report.recommendations:
            print("\nRecommendations:")

            for recommendation in (
                final_report.recommendations
            ):
                print(
                    f"- ["
                    f"{recommendation.priority.upper()}"
                    f"] {recommendation.action}"
                )

                print(
                    f"  Rationale: "
                    f"{recommendation.rationale}"
                )

        if final_report.chart is not None:
            print("\nChart specification:")

            print(
                f"Type                          : "
                f"{final_report.chart.chart_type}"
            )

            print(
                f"Title                         : "
                f"{final_report.chart.title}"
            )

            print(
                f"X column                      : "
                f"{final_report.chart.x_column}"
            )

            print(
                f"Y columns                     : "
                f"{final_report.chart.y_columns}"
            )

            print(
                f"Category                      : "
                f"{final_report.chart.category_column}"
            )

            print(
                f"Rationale                     : "
                f"{final_report.chart.rationale}"
            )

        if final_report.limitations:
            print("\nReport limitations:")

            for limitation in (
                final_report.limitations
            ):
                print(f"- {limitation}")

    deterministic_result = final_state.get(
        "deterministic_validation_result"
    )

    if deterministic_result is not None:
        print(
            "\nDeterministic output validation:"
        )

        print(
            f"Decision                      : "
            f"{deterministic_result.decision}"
        )

        print(
            f"Valid                         : "
            f"{deterministic_result.valid}"
        )

        print(
            f"Numerical claims valid        : "
            f"{deterministic_result.numerical_claims_valid}"
        )

        print(
            f"Ranking valid                 : "
            f"{deterministic_result.ranking_valid}"
        )

        print(
            f"Chart valid                   : "
            f"{deterministic_result.chart_valid}"
        )

        print(
            f"PII safe                      : "
            f"{deterministic_result.pii_safe}"
        )

        print(
            f"Secret safe                   : "
            f"{deterministic_result.secret_safe}"
        )

        print(
            f"Prompt leakage safe           : "
            f"{deterministic_result.prompt_leakage_safe}"
        )

        if deterministic_result.issues:
            print(
                "\nDeterministic validation issues:"
            )

            for issue in (
                deterministic_result.issues
            ):
                print(f"- {issue}")

    output_evaluation = final_state.get(
        "output_evaluation"
    )

    if output_evaluation is not None:
        print("\nLLM output evaluation:")

        print(
            f"Decision                      : "
            f"{output_evaluation.decision}"
        )

        print(
            f"Relevance score               : "
            f"{output_evaluation.relevance_score:.2f}"
        )

        print(
            f"Groundedness score            : "
            f"{output_evaluation.groundedness_score:.2f}"
        )

        print(
            f"Accuracy score                : "
            f"{output_evaluation.accuracy_score:.2f}"
        )

        print(
            f"Safety score                  : "
            f"{output_evaluation.safety_score:.2f}"
        )

        print(
            f"Overall score                 : "
            f"{output_evaluation.overall_score:.2f}"
        )

        if output_evaluation.issues:
            print(
                "\nLLM evaluation issues:"
            )

            for issue in (
                output_evaluation.issues
            ):
                print(f"- {issue}")

    output_guard_result = final_state.get(
        "output_guard_result"
    )

    if output_guard_result is not None:
        print("\nFinal output guard:")

        print(
            f"Decision                      : "
            f"{output_guard_result.decision}"
        )

        print(
            f"Redactions                    : "
            f"{len(output_guard_result.redactions)}"
        )

        print(
            f"Secret data                   : "
            f"{output_guard_result.contains_secret_data}"
        )

        print(
            f"Prompt leakage                : "
            f"{output_guard_result.contains_prompt_leakage}"
        )

        print(
            f"Unauthorized fields           : "
            f"{output_guard_result.contains_unauthorized_fields}"
        )

        print(
            f"Sanitized payload available   : "
            f"{output_guard_result.sanitized_payload is not None}"
        )

        print(
            f"Guard message                 : "
            f"{output_guard_result.user_message}"
        )

    output_validation_message = (
        final_state.get(
            "output_validation_message"
        )
    )

    if output_validation_message:
        print(
            f"\nOutput validation message: "
            f"{output_validation_message}"
        )

    error_type = final_state.get(
        "error_type"
    )

    error_message = final_state.get(
        "error_message"
    )

    if error_type or error_message:
        print("\nWorkflow error:")

        print(
            f"Type                          : "
            f"{error_type}"
        )

        print(
            f"Message                       : "
            f"{error_message}"
        )

    print("\nExecution trace:")

    for event in final_state.get(
        "execution_trace",
        [],
    ):
        print(
            f"- {event['node']} | "
            f"{event['status']} | "
            f"{event['message']}"
        )


# ---------------------------------------------------------------------
# Standalone complete-workflow smoke test
# ---------------------------------------------------------------------

def main() -> None:
    """
    Run the complete security-enabled multi-agent workflow.

    Expected successful final status:

        completed

    Controlled terminal statuses:

        clarification_required
        security_blocked
        security_review_required
        output_blocked
        output_review_required
        failed
    """

    print("=" * 80)
    print(
        "COMPLETE SECURITY-ENABLED "
        "LANGGRAPH WORKFLOW"
    )
    print("=" * 80)

    question = (
        "Show the top 5 products by revenue."
    )

    print(
        f"Question: {question}"
    )

    print("\nRunning workflow...")

    try:
        final_state = run_workflow(
            user_question=question,
            max_sql_retries=3,
            max_report_regenerations=1,
            recursion_limit=50,
        )

        print_workflow_result(
            final_state
        )

        workflow_status = (
            final_state.get(
                "workflow_status"
            )
        )

        # -------------------------------------------------------------
        # Technical failure
        # -------------------------------------------------------------

        if workflow_status == "failed":
            raise RuntimeError(
                final_state.get(
                    "error_message",
                    "Workflow failed.",
                )
            )

        # -------------------------------------------------------------
        # Input security outcomes
        # -------------------------------------------------------------

        if (
            workflow_status
            == "security_blocked"
        ):
            print("\n" + "=" * 80)
            print(
                "RESULT: REQUEST BLOCKED BY "
                "INPUT SECURITY POLICY"
            )
            print("=" * 80)

            print(
                final_state.get(
                    "security_message",
                    (
                        "The request cannot be processed "
                        "because it violates security policy."
                    ),
                )
            )

            return

        if (
            workflow_status
            == "security_review_required"
        ):
            print("\n" + "=" * 80)
            print(
                "RESULT: INPUT SECURITY "
                "REVIEW REQUIRED"
            )
            print("=" * 80)

            print(
                final_state.get(
                    "security_message",
                    (
                        "The request requires review "
                        "before processing."
                    ),
                )
            )

            return

        # -------------------------------------------------------------
        # Output-control outcomes
        # -------------------------------------------------------------

        if (
            workflow_status
            == "output_blocked"
        ):
            print("\n" + "=" * 80)
            print(
                "RESULT: GENERATED OUTPUT "
                "BLOCKED"
            )
            print("=" * 80)

            print(
                final_state.get(
                    "output_validation_message",
                    (
                        "The generated response was blocked "
                        "by output security controls."
                    ),
                )
            )

            return

        if (
            workflow_status
            == "output_review_required"
        ):
            print("\n" + "=" * 80)
            print(
                "RESULT: OUTPUT QUALITY OR "
                "SECURITY REVIEW REQUIRED"
            )
            print("=" * 80)

            print(
                final_state.get(
                    "output_validation_message",
                    (
                        "The generated response requires "
                        "quality or security review."
                    ),
                )
            )

            return

        # -------------------------------------------------------------
        # Clarification outcome
        # -------------------------------------------------------------

        if final_state.get(
            "clarification_required",
            False,
        ):
            print("\n" + "=" * 80)
            print(
                "RESULT: WORKFLOW REQUIRES "
                "CLARIFICATION"
            )
            print("=" * 80)

            return

        # -------------------------------------------------------------
        # Successful output-contract checks
        # -------------------------------------------------------------

        if final_state.get(
            "intent"
        ) is None:
            raise RuntimeError(
                "The workflow did not produce "
                "IntentResult."
            )

        if final_state.get(
            "plan"
        ) is None:
            raise RuntimeError(
                "The workflow did not produce "
                "AnalysisPlan."
            )

        if not final_state.get(
            "database_schema"
        ):
            raise RuntimeError(
                "The workflow did not prepare "
                "the database schema."
            )

        if not final_state.get(
            "generated_sql"
        ):
            raise RuntimeError(
                "Agent 3a did not generate SQL."
            )

        if final_state.get(
            "query_result"
        ) is None:
            raise RuntimeError(
                "Agent 3b did not return "
                "QueryExecutionResult."
            )

        if final_state.get(
            "analysis_result"
        ) is None:
            raise RuntimeError(
                "Agent 4 did not return "
                "AnalysisResult."
            )

        if final_state.get(
            "final_report"
        ) is None:
            raise RuntimeError(
                "Agent 5 did not return "
                "FinalReport."
            )

        if final_state.get(
            "deterministic_validation_result"
        ) is None:
            raise RuntimeError(
                "The workflow did not produce "
                "DeterministicValidationResult."
            )

        if final_state.get(
            "output_evaluation"
        ) is None:
            raise RuntimeError(
                "The workflow did not produce "
                "OutputEvaluation."
            )

        if final_state.get(
            "output_guard_result"
        ) is None:
            raise RuntimeError(
                "The workflow did not produce "
                "OutputGuardResult."
            )

        if final_state.get(
            "sanitized_output_payload"
        ) is None:
            raise RuntimeError(
                "The workflow did not produce a "
                "sanitized output payload."
            )

        if workflow_status != "completed":
            raise RuntimeError(
                "Expected workflow status "
                f"'completed', received "
                f"'{workflow_status}'."
            )

        print("\n" + "=" * 80)
        print(
            "RESULT: COMPLETE SECURITY-ENABLED "
            "WORKFLOW IS WORKING"
        )
        print("=" * 80)

        print(
            "Input security, prompt-risk evaluation, "
            "intent understanding, planning, schema "
            "preparation, SQL generation, SQL validation, "
            "privacy-policy enforcement, read-only execution, "
            "analysis, reporting, deterministic validation, "
            "LLM evaluation, and final output protection "
            "are integrated."
        )

    except Exception as exc:
        print("\n" + "=" * 80)
        print(
            "RESULT: COMPLETE SECURITY-ENABLED "
            "WORKFLOW FAILED"
        )
        print("=" * 80)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()