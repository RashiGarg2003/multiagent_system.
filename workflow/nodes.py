"""
LangGraph nodes for the Multi-Agent Data Analytics System.

The workflow includes checkpointed clarification, read-only MCP database
access, Agents 1 through 5, and one unified post-report evaluator.

Each node:

- Reads values from GraphState.
- Calls an existing project component.
- Returns a partial GraphState update.
- Adds a safe execution-trace event.
- Handles errors without exposing secrets or database rows.
"""

from typing import Any, Optional
from agents.analysis_agent import AnalysisAgent
from agents.intent_agent import IntentAgent
from agents.planning_agent import PlanningAgent
from agents.reporting_agent import ReportingAgent
from agents.sql_agent import SQLAgent
from security.output_guard import OutputGuard

from models.schemas import (
    AnalysisPlan,
    IntentResult,
    QueryExecutionResult,
)

from tools.sql_execution_tool import (
    SQLExecutionOutcome,
)
from security.input_guard import InputGuard
from security.prompt_injection_guard import (
    PromptInjectionGuard,
)
from security.models import (
    InputGuardResult,
    PromptRiskAssessment,
    UserRole,
)


from evaluation.deterministic_validator import (
    DeterministicOutputValidator,
)
from evaluation.llm_evaluator import (
    LLMOutputEvaluator,
)
from evaluation.unified_evaluator import (
    UnifiedOutputEvaluator,
)

from workflow.state import (
    GraphState,
    add_trace_event,
)
from mcp_layer.client import InternalMCPClient
from langgraph.types import interrupt

def enum_value(
    value: Any,
) -> Any:
    """
    Return the primitive value stored in an Enum.

    Examples:

        EvaluationDecision.APPROVE -> "approve"
        SecurityDecision.ALLOW -> "allow"
        "approve" -> "approve"
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


class WorkflowNodes:
    """
    Collection of all LangGraph workflow nodes.

    Existing agents and tools are injected into this class so that
    LangGraph does not create duplicate Ollama clients or database
    connections for every node.
    """

    def __init__(
    self,
    input_guard: InputGuard,
    prompt_injection_guard: PromptInjectionGuard,
    intent_agent: IntentAgent,
    planning_agent: PlanningAgent,
    sql_agent: SQLAgent,
    analysis_agent: AnalysisAgent,
    reporting_agent: ReportingAgent,
    unified_output_evaluator: UnifiedOutputEvaluator,
    mcp_client: InternalMCPClient,
) -> None:

        """
        Initialize workflow nodes using existing project components.
        """
        if not isinstance(
            input_guard,
            InputGuard,
        ):
            raise TypeError(
                "input_guard must be an "
                "InputGuard instance."
            )

        if not isinstance(
            prompt_injection_guard,
            PromptInjectionGuard,
        ):
            raise TypeError(
                "prompt_injection_guard must be a "
                "PromptInjectionGuard instance."
            )


        if not isinstance(
            intent_agent,
            IntentAgent,
        ):
            raise TypeError(
                "intent_agent must be an IntentAgent instance."
            )

        if not isinstance(
            planning_agent,
            PlanningAgent,
        ):
            raise TypeError(
                "planning_agent must be a PlanningAgent instance."
            )

        if not isinstance(
            sql_agent,
            SQLAgent,
        ):
            raise TypeError(
                "sql_agent must be a SQLAgent instance."
            )

        if not isinstance(
            analysis_agent,
            AnalysisAgent,
                ):
                    raise TypeError(
                "analysis_agent must be an "
                "AnalysisAgent instance."
            )

        if not isinstance(
            reporting_agent,
            ReportingAgent,
                ):
                    raise TypeError(
                "reporting_agent must be a "
                "ReportingAgent instance."
            )   

        if not isinstance(
            unified_output_evaluator,
            UnifiedOutputEvaluator,
        ):
            raise TypeError(
                "unified_output_evaluator must be a "
                "UnifiedOutputEvaluator instance."
            )

        if not isinstance(mcp_client, InternalMCPClient):
            raise TypeError(
                "mcp_client must be an InternalMCPClient instance."
            )

        self.input_guard = input_guard
        self.prompt_injection_guard = (
            prompt_injection_guard
        )

        self.intent_agent = intent_agent
        self.planning_agent = planning_agent
        self.sql_agent = sql_agent
        self.analysis_agent = analysis_agent
        self.reporting_agent = reporting_agent

        self.unified_output_evaluator = (
            unified_output_evaluator
        )

        self.mcp_client = mcp_client

    def _get_original_question(
        self,
        state: GraphState,
    ) -> str:
        """
        Return the original unsanitized question.

        This value is used only by the deterministic input guard. It must
        not be written to the execution trace or unrestricted API output.
        """

        question = state.get(
            "original_user_question"
        )

        if not isinstance(
            question,
            str,
        ):
            raise ValueError(
                "GraphState does not contain a valid "
                "original_user_question."
            )

        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError(
                "GraphState original_user_question "
                "cannot be empty."
            )

        return cleaned_question

    def _get_required_question(
        self,
        state: GraphState,
    ) -> str:
        """
        Return the validated user question from GraphState.
        """

        question = state.get(
            "user_question"
        )

        if not isinstance(
            question,
            str,
        ):
            raise ValueError(
                "GraphState does not contain a valid "
                "user_question."
            )

        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError(
                "GraphState user_question cannot be empty."
            )

        return cleaned_question

    def _get_required_intent(
        self,
        state: GraphState,
    ) -> IntentResult:
        """
        Return the validated IntentResult from GraphState.
        """

        intent = state.get("intent")

        if not isinstance(
            intent,
            IntentResult,
        ):
            raise ValueError(
                "GraphState does not contain a valid "
                "IntentResult."
            )

        return intent

    def _get_required_plan(
        self,
        state: GraphState,
    ) -> AnalysisPlan:
        """
        Return the validated AnalysisPlan from GraphState.
        """

        plan = state.get("plan")

        if not isinstance(
            plan,
            AnalysisPlan,
        ):
            raise ValueError(
                "GraphState does not contain a valid "
                "AnalysisPlan."
            )

        if not plan.steps:
            raise ValueError(
                "The AnalysisPlan contains no steps."
            )

        return plan

    def _get_required_schema(
        self,
        state: GraphState,
    ) -> str:
        """
        Return the prompt-ready database schema.
        """

        database_schema = state.get(
            "database_schema"
        )

        if not isinstance(
            database_schema,
            str,
        ):
            raise ValueError(
                "GraphState does not contain a valid "
                "database schema."
            )

        cleaned_schema = (
            database_schema.strip()
        )

        if not cleaned_schema:
            raise ValueError(
                "GraphState database schema is empty."
            )

        return cleaned_schema

    def _error_update(
        self,
        state: GraphState,
        node_name: str,
        error: Exception,
        mcp_tool: str | None = None,
    ) -> dict[str, Any]:
        """
        Build a controlled workflow failure update.

        Prompts, secrets, complete schema text, and database result rows
        are not added to the execution trace.
        """

        error_type = type(error).__name__

        trace = add_trace_event(
            state=state,
            node=node_name,
            status="failed",
            message=(
                f"{node_name} failed with "
                f"{error_type}."
            ),
            mcp_tool=mcp_tool,
        )

        return {
            "workflow_status": "failed",
            "current_node": node_name,
            "error_type": error_type,
            "error_message": str(error),
            "execution_trace": trace,
        }

    def input_guard_node(
            self,
            state: GraphState,
        ) -> dict[str, Any]:
            """
            Execute deterministic input-security screening.

            Reads:
                original_user_question

            Writes:
                input_guard_result
                user_question
                question_was_redacted
                security_decision
                security_message
                security_review_required
                security_risk_categories
                workflow_status
                current_node
                execution_trace

            Clear destructive, privacy, secret-extraction, filesystem, and
            other prohibited requests are stopped before Agent 1.
            """

            node_name = "input_guard"

            try:
                original_question = (
                    self._get_original_question(
                        state
                    )
                )

                guard_result = (
                    self.input_guard.evaluate(
                        original_question
                    )
                )

                if not isinstance(
                    guard_result,
                    InputGuardResult,
                ):
                    raise TypeError(
                        "InputGuard did not return "
                        "InputGuardResult."
                    )

                decision = str(
                    guard_result.decision
                )

                if decision == "block":
                    workflow_status = (
                        "security_blocked"
                    )

                    safe_question = (
                        state.get(
                            "user_question",
                            original_question,
                        )
                    )

                    trace_status = "blocked"

                    trace_message = (
                        "The request was blocked by "
                        "deterministic input-security controls."
                    )

                elif decision == "review":
                    workflow_status = (
                        "security_review_required"
                    )

                    safe_question = (
                        guard_result.sanitized_question
                        or state.get(
                            "user_question",
                            original_question,
                        )
                    )

                    trace_status = (
                        "review_required"
                    )

                    trace_message = (
                        "The request requires security review "
                        "before analytical processing."
                    )

                else:
                    safe_question = (
                        guard_result.sanitized_question
                    )

                    if not isinstance(
                        safe_question,
                        str,
                    ) or not safe_question.strip():
                        raise RuntimeError(
                            "InputGuard approved the request "
                            "without a sanitized question."
                        )

                    safe_question = (
                        safe_question.strip()
                    )

                    workflow_status = (
                        "evaluating_prompt_risk"
                    )

                    trace_status = "completed"

                    if decision == "allow_redacted":
                        trace_message = (
                            "Input security screening passed "
                            "after sensitive values were redacted."
                        )

                    else:
                        trace_message = (
                            "Input security screening passed."
                        )

                question_was_redacted = (
                    isinstance(safe_question, str)
                    and safe_question
                    != original_question
                )

                trace = add_trace_event(
                    state=state,
                    node=node_name,
                    status=trace_status,
                    message=trace_message,
                )

                return {
                    "input_guard_result": (
                        guard_result
                    ),
                    "user_question": safe_question,
                    "question_was_redacted": (
                        question_was_redacted
                    ),
                    "security_decision": decision,
                    "security_message": (
                        guard_result.user_message
                    ),
                    "security_review_required": (
                        decision == "review"
                    ),
                    "security_risk_categories": list(
                        guard_result.risk_categories
                    ),
                    "workflow_status": (
                        workflow_status
                    ),
                    "current_node": node_name,
                    "error_type": None,
                    "error_message": None,
                    "execution_trace": trace,
                }

            except Exception as exc:
                return self._error_update(
                    state=state,
                    node_name=node_name,
                    error=exc,
                )

    def prompt_injection_guard_node(
            self,
            state: GraphState,
        ) -> dict[str, Any]:
            """
            Execute semantic prompt-risk classification.

            Only the sanitized question is sent to the local LLM.

            Reads:
                user_question
                input_guard_result

            Writes:
                prompt_risk_assessment
                security_decision
                security_message
                security_review_required
                workflow_status
                current_node
                execution_trace
            """

            node_name = "prompt_injection_guard"

            try:
                question = self._get_required_question(
                    state
                )

                input_guard_result = state.get(
                    "input_guard_result"
                )

                if not isinstance(
                    input_guard_result,
                    InputGuardResult,
                ):
                    raise ValueError(
                        "Prompt-risk evaluation requires "
                        "InputGuardResult."
                    )

                deterministic_decision = str(
                    input_guard_result.decision
                )

                if deterministic_decision in {
                    "block",
                    "review",
                }:
                    raise ValueError(
                        "Prompt-risk evaluation cannot run "
                        "after a blocking or review decision."
                    )

                assessment = (
                    self.prompt_injection_guard.assess(
                        question=question,
                        deterministic_result=(
                            input_guard_result
                        ),
                    )
                )

                if not isinstance(
                    assessment,
                    PromptRiskAssessment,
                ):
                    raise TypeError(
                        "PromptInjectionGuard did not return "
                        "PromptRiskAssessment."
                    )

                semantic_decision = str(
                    assessment.decision
                )

                if semantic_decision == "block":
                    final_decision = "block"

                    workflow_status = (
                        "security_blocked"
                    )

                    security_message = (
                        "The request was blocked because "
                        "semantic security screening detected "
                        "an unsafe instruction."
                    )

                    trace_status = "blocked"

                    trace_message = (
                        "The request was blocked by semantic "
                        "prompt-risk screening."
                    )

                elif semantic_decision == "review":
                    final_decision = "review"

                    workflow_status = (
                        "security_review_required"
                    )

                    security_message = (
                        "The request requires security review "
                        "before analytical processing."
                    )

                    trace_status = (
                        "review_required"
                    )

                    trace_message = (
                        "Semantic prompt-risk screening "
                        "requires human review."
                    )

                else:
                    # Preserve the fact that deterministic PII screening
                    # redacted the question.
                    if (
                        deterministic_decision
                        == "allow_redacted"
                    ):
                        final_decision = (
                            "allow_redacted"
                        )

                    else:
                        final_decision = "allow"

                    workflow_status = (
                        "understanding_intent"
                    )

                    security_message = (
                        "The request passed input and "
                        "prompt-risk screening."
                    )

                    trace_status = "completed"

                    trace_message = (
                        "Semantic prompt-risk screening passed."
                    )

                trace = add_trace_event(
                    state=state,
                    node=node_name,
                    status=trace_status,
                    message=trace_message,
                )

                return {
                    "prompt_risk_assessment": (
                        assessment
                    ),
                    "security_decision": (
                        final_decision
                    ),
                    "security_message": (
                        security_message
                    ),
                    "security_review_required": (
                        final_decision == "review"
                    ),
                    "workflow_status": (
                        workflow_status
                    ),
                    "current_node": node_name,
                    "error_type": None,
                    "error_message": None,
                    "execution_trace": trace,
                }

            except Exception as exc:
                return self._error_update(
                    state=state,
                    node_name=node_name,
                    error=exc,
                )

    def intent_node(
        self,
        state: GraphState,
    ) -> dict[str, Any]:
        """
        Execute Agent 1: Intent Understanding.

        Reads:
            user_question

        Writes:
            intent
            clarification_required
            clarification_question
            workflow_status
            current_node
            execution_trace

        A request proceeds to planning only when the structured intent
        contains sufficient analytical context. Otherwise, the workflow
        asks the user for clarification before SQL generation.
        """

        node_name = "intent_agent"

        try:
            question = (
                self._get_required_question(
                    state
                )
            )

            intent = (
                self.intent_agent.understand(
                    question
                )
            )

            if intent is None:
                raise RuntimeError(
                    "IntentAgent returned no IntentResult."
                )

            # ---------------------------------------------------------
            # Deterministic clarification safeguard
            # ---------------------------------------------------------
            #
            # Do not depend only on the LLM's clarification decision.
            # Verify whether the structured intent contains enough context
            # for planning and SQL generation.
            #
            # This is generalizable because it evaluates the Pydantic
            # output fields rather than matching fixed user phrases.

            has_metric = bool(
                [
                    metric
                    for metric in intent.metrics
                    if str(metric).strip()
                ]
            )

            has_dimension = bool(
                [
                    dimension
                    for dimension in intent.dimensions
                    if str(dimension).strip()
                ]
            )

            has_filter = bool(
                intent.filters
            )

            has_actionable_intent = any(
                [
                    has_metric,
                    has_dimension,
                    has_filter,
                ]
            )

            # ---------------------------------------------------------
            # Force clarification for an underspecified request
            # ---------------------------------------------------------

            if (
                not intent.clarification_required
                and not has_actionable_intent
            ):
                intent = intent.model_copy(
                    update={
                        "clarification_required": True,
                        "clarification_question": (
                            "What metric would you like to analyze, "
                            "and how should the results be grouped, "
                            "filtered, or compared?"
                        ),
                    }
                )

            # Ensure that clarification always contains a useful question.
            if intent.clarification_required:
                clarification_question = (
                    intent.clarification_question
                )

                if (
                    not isinstance(
                        clarification_question,
                        str,
                    )
                    or not clarification_question.strip()
                ):
                    clarification_question = (
                        "What metric would you like to analyze, "
                        "and how should the results be grouped, "
                        "filtered, or compared?"
                    )

                    intent = intent.model_copy(
                        update={
                            "clarification_question": (
                                clarification_question
                            ),
                        }
                    )

                next_status = (
                    "clarification_required"
                )

                trace_status = (
                    "clarification_required"
                )

                trace_message = (
                    "Agent 1 requires additional "
                    "analytical context."
                )

            else:
                next_status = "planning"

                trace_status = "completed"

                trace_message = (
                    "Intent understood and validated."
                )

            trace = add_trace_event(
                state=state,
                node=node_name,
                status=trace_status,
                message=trace_message,
            )

            return {
                "intent": intent,
                "clarification_required": (
                    intent.clarification_required
                ),
                "clarification_question": (
                    intent.clarification_question
                ),
                "workflow_status": (
                    next_status
                ),
                "current_node": node_name,
                "error_type": None,
                "error_message": None,
                "execution_trace": trace,
            }

        except Exception as exc:
            error_update = self._error_update(
                state=state,
                node_name=node_name,
                error=exc,
            )
            return error_update
    
    def clarification_node(
            self,
            state: GraphState,
        ) -> dict[str, Any]:
            """
            Pause the current thread and screen the resumed answer.
            """

            node_name = "clarification"

            clarification_question = state.get(
                "clarification_question"
            )
            if (
                not isinstance(clarification_question, str)
                or not clarification_question.strip()
            ):
                clarification_question = (
                    "What metric would you like to analyze, "
                    "and how should the results be grouped or filtered?"
                )

            answer = interrupt(
                {
                    "question": clarification_question,
                    "message": (
                        "Additional analytical context is required."
                    ),
                }
            )
            if not isinstance(answer, str) or not answer.strip():
                return self._error_update(
                    state=state,
                    node_name=node_name,
                    error=ValueError(
                        "A non-empty clarification answer is required."
                    ),
                )

            answer_guard = self.input_guard.evaluate(answer.strip())
            answer_decision = enum_value(answer_guard.decision)
            if answer_decision in {"block", "review"}:
                status = (
                    "security_blocked"
                    if answer_decision == "block"
                    else "security_review_required"
                )
                trace = add_trace_event(
                    state=state,
                    node=node_name,
                    status=answer_decision,
                    message=(
                        "The clarification answer did not pass "
                        "input-security screening."
                    ),
                )
                return {
                    "input_guard_result": answer_guard,
                    "security_decision": answer_decision,
                    "security_message": answer_guard.user_message,
                    "security_review_required": answer_decision == "review",
                    "workflow_status": status,
                    "current_node": node_name,
                    "error_type": None,
                    "error_message": None,
                    "execution_trace": trace,
                }

            safe_answer = answer_guard.sanitized_question
            if not isinstance(safe_answer, str) or not safe_answer.strip():
                raise RuntimeError(
                    "Input security screening returned no safe clarification."
                )
            safe_answer = safe_answer.strip()
            prompt_assessment = self.prompt_injection_guard.assess(
                question=safe_answer,
                deterministic_result=answer_guard,
            )
            prompt_decision = enum_value(prompt_assessment.decision)
            if prompt_decision in {"block", "review"}:
                status = (
                    "security_blocked"
                    if prompt_decision == "block"
                    else "security_review_required"
                )
                trace = add_trace_event(
                    state=state,
                    node=node_name,
                    status=prompt_decision,
                    message=(
                        "The clarification answer did not pass "
                        "prompt-risk screening."
                    ),
                )
                return {
                    "input_guard_result": answer_guard,
                    "prompt_risk_assessment": prompt_assessment,
                    "security_decision": prompt_decision,
                    "security_message": (
                        "The clarification answer requires security "
                        "review before analytical processing."
                    ),
                    "security_review_required": prompt_decision == "review",
                    "workflow_status": status,
                    "current_node": node_name,
                    "error_type": None,
                    "error_message": None,
                    "execution_trace": trace,
                }

            current_question = self._get_required_question(state)
            combined_question = (
                f"{current_question}\n\nAdditional context: {safe_answer}"
            )
            trace = add_trace_event(
                state=state,
                node=node_name,
                status="resumed",
                message=(
                    "The clarification answer passed security screening; "
                    "intent analysis is resuming."
                ),
            )
            return {
                "user_question": combined_question,
                "input_guard_result": answer_guard,
                "prompt_risk_assessment": prompt_assessment,
                "security_decision": (
                    "allow_redacted"
                    if answer_decision == "allow_redacted"
                    else "allow"
                ),
                "security_message": (
                    "The clarification answer passed security screening."
                ),
                "security_review_required": False,
                "intent": None,
                "clarification_required": False,
                "clarification_question": None,
                "workflow_status": "understanding_intent",
                "current_node": node_name,
                "error_type": None,
                "error_message": None,
                "execution_trace": trace,
            }

    def planning_node(
        self,
        state: GraphState,
    ) -> dict[str, Any]:
        """
        Execute Agent 2: Planning.

        Reads:
            intent

        Writes:
            plan
            workflow_status
            execution_trace
        """

        node_name = "planning_agent"

        try:
            intent = self._get_required_intent(
                state
            )

            if intent.clarification_required:
                raise ValueError(
                    "Planning cannot begin while "
                    "clarification is required."
                )

            plan = (
                self.planning_agent.create_plan(
                    intent=intent
                )
            )

            trace = add_trace_event(
                state=state,
                node=node_name,
                status="completed",
                message=(
                    "Analytical plan created and "
                    "validated."
                ),
            )

            return {
                "plan": plan,
                "workflow_status": (
                    "generating_sql"
                ),
                "current_node": node_name,
                "error_type": None,
                "error_message": None,
                "execution_trace": trace,
            }

        except Exception as exc:
            return self._error_update(
                state=state,
                node_name=node_name,
                error=exc,
            )

    def schema_node(
        self,
        state: GraphState,
    ) -> dict[str, Any]:
        """
        Prepare the actual SQLite schema for Agent 3a.

        The database schema is extracted dynamically. No table or
        column names are hardcoded.
        """

        node_name = "schema_manager"
        mcp_tool_called = False

        try:
            mcp_tool_called = True
            result = self.mcp_client.call_tool(
                "get_approved_schema",
                {},
            )
            database_schema = result.get("schema")

            if not isinstance(database_schema, str) or not database_schema.strip():
                raise RuntimeError(
                    "The approved schema tool returned no schema."
                )
            cleaned_schema = database_schema.strip()

            trace = add_trace_event(
                state=state,
                node=node_name,
                status="completed",
                message=(
                    "Database verified and schema "
                    "prepared for SQL generation."
                ),
                mcp_tool="get_approved_schema",
            )

            return {
                "database_schema": cleaned_schema,
                "workflow_status": (
                    "generating_sql"
                ),
                "current_node": node_name,
                "error_type": None,
                "error_message": None,
                "execution_trace": trace,
            }

        except Exception as exc:
            return self._error_update(
                state=state,
                node_name=node_name,
                error=exc,
                mcp_tool=(
                    "get_approved_schema"
                    if mcp_tool_called
                    else None
                ),
            )

    def _build_sql_retry_feedback(
        self,
        state: GraphState,
    ) -> Optional[str]:
        """
        Combine Agent 3b validation and database errors.

        This feedback is returned to Agent 3a during a retry.
        """

        feedback: list[str] = []

        validation_errors = state.get(
            "sql_validation_errors",
            [],
        )

        if isinstance(
            validation_errors,
            list,
        ):
            for error in validation_errors:
                if (
                    isinstance(error, str)
                    and error.strip()
                ):
                    feedback.append(
                        error.strip()
                    )

        execution_error = state.get(
            "sql_execution_error"
        )

        if (
            isinstance(execution_error, str)
            and execution_error.strip()
        ):
            feedback.append(
                execution_error.strip()
            )

        if not feedback:
            return None

        return "\n".join(
            f"- {error}"
            for error in feedback
        )

    def sql_generation_node(
        self,
        state: GraphState,
    ) -> dict[str, Any]:
        """
        Execute Agent 3a: SQL Generation.

        On the first attempt, Agent 3a receives:

        - Question
        - Intent
        - Plan
        - Actual schema

        During retry, Agent 3a additionally receives:

        - Previous SQL
        - Agent 3b validation error
        - Actual SQLite compilation or execution error
        """

        node_name = "sql_generation_agent"

        try:
            question = (
                self._get_required_question(
                    state
                )
            )

            intent = self._get_required_intent(
                state
            )

            if intent.clarification_required:
                raise ValueError(
                    "SQL generation cannot begin while "
                    "clarification is required."
                )

            plan = self._get_required_plan(
                state
            )

            database_schema = (
                self._get_required_schema(
                    state
                )
            )

            retry_count = state.get(
                "sql_retry_count",
                0,
            )

            if not isinstance(
                retry_count,
                int,
            ):
                raise ValueError(
                    "sql_retry_count must be "
                    "an integer."
                )

            previous_sql: Optional[str] = None
            previous_error: Optional[str] = None

            if retry_count > 0:
                stored_sql = state.get(
                    "generated_sql"
                )

                if (
                    isinstance(stored_sql, str)
                    and stored_sql.strip()
                ):
                    previous_sql = (
                        stored_sql.strip()
                    )

                previous_error = (
                    self._build_sql_retry_feedback(
                        state
                    )
                )

                if previous_error is None:
                    raise ValueError(
                        "SQL retry was requested but "
                        "no Agent 3b error feedback "
                        "is available."
                    )

            sql_result = (
                self.sql_agent.generate_sql(
                    user_question=question,
                    intent=intent,
                    plan=plan,
                    database_schema=database_schema,
                    previous_sql=previous_sql,
                    previous_error=previous_error,
                )
            )

            generated_sql = (
                sql_result.sql.strip()
            )

            if not generated_sql:
                raise RuntimeError(
                    "Agent 3a returned empty SQL."
                )

            if retry_count > 0:
                trace_message = (
                    "Agent 3a generated corrected SQL "
                    f"for retry {retry_count}."
                )
            else:
                trace_message = (
                    "Agent 3a generated one read-only "
                    "SQLite query."
                )

            trace = add_trace_event(
                state=state,
                node=node_name,
                status="completed",
                message=trace_message,
            )

            return {
                "sql_result": sql_result,
                "generated_sql": generated_sql,

                # The previous error has been consumed by Agent 3a.
                # Agent 3b will populate these fields again if the
                # corrected query still fails.
                "sql_validation_errors": [],
                "sql_execution_error": None,
                "query_result": None,

                "workflow_status": (
                    "validating_sql"
                ),
                "current_node": node_name,
                "error_type": None,
                "error_message": None,
                "execution_trace": trace,
            }

        except Exception as exc:
            return self._error_update(
                state=state,
                node_name=node_name,
                error=exc,
            )

    def _sql_retry_update(
        self,
        state: GraphState,
        outcome: SQLExecutionOutcome,
    ) -> dict[str, Any]:
        """
        Return a retry update after Agent 3b rejects SQL.
        """

        node_name = (
            "sql_validation_execution"
        )

        retry_count = state.get(
            "sql_retry_count",
            0,
        )

        if not isinstance(
            retry_count,
            int,
        ):
            retry_count = 0

        next_retry_count = (
            retry_count + 1
        )

        trace = add_trace_event(
            state=state,
            node=node_name,
            status="retry_required",
            message=(
                "Agent 3b rejected SQL during "
                f"'{outcome.stage}'. "
                f"Retry count is {next_retry_count}."
            ),
            mcp_tool="execute_validated_read_query",
        )

        return {
            "query_result": None,
            "sql_validation_errors": list(
                outcome.validation_errors
            ),
            "sql_execution_error": (
                outcome.execution_error
            ),
            "sql_retry_count": (
                next_retry_count
            ),
            "workflow_status": (
                "retrying_sql"
            ),
            "current_node": node_name,
            "error_type": None,
            "error_message": None,
            "execution_trace": trace,
        }

    def sql_validation_execution_node(
        self,
        state: GraphState,
    ) -> dict[str, Any]:
        """
        Execute Agent 3b: SQL Validation and Execution.

        Validation sequence:

        1. SQL AST safety validation
        2. Approved table and view validation
        3. SQLite EXPLAIN QUERY PLAN
        4. Read-only database execution
        5. QueryExecutionResult validation
        """

        node_name = (
            "sql_validation_execution"
        )
        mcp_tool_called = False

        try:
            generated_sql = state.get(
                "generated_sql"
            )

            if not isinstance(
                generated_sql,
                str,
            ):
                raise ValueError(
                    "GraphState does not contain "
                    "generated SQL."
                )

            cleaned_sql = (
                generated_sql.strip()
            )

            if not cleaned_sql:
                raise ValueError(
                    "GraphState generated SQL is empty."
                )

            intent = self._get_required_intent(state)
            plan = self._get_required_plan(state)
            mcp_tool_called = True
            tool_result = self.mcp_client.call_tool(
                "execute_validated_read_query",
                {
                    "sql": cleaned_sql,
                    "intent_json": intent.model_dump_json(),
                    "plan_json": plan.model_dump_json(),
                },
            )

            if not isinstance(tool_result, dict):
                raise TypeError(
                    "The validated-query MCP tool returned an invalid result."
                )

            if tool_result.get("success") is not True:
                retryable = tool_result.get("retryable") is True
                outcome = SQLExecutionOutcome(
                    success=False,
                    validated_sql=None,
                    validation_result=None,
                    query_result=None,
                    validation_errors=list(
                        tool_result.get("validation_errors", [])
                    ),
                    execution_error=tool_result.get("execution_error"),
                    retryable=retryable,
                    stage=str(tool_result.get("stage", "validation")),
                )
                if retryable:
                    return self._sql_retry_update(
                        state=state,
                        outcome=outcome,
                    )
                if tool_result.get("security_violation") is True:
                    trace = add_trace_event(
                        state=state,
                        node=node_name,
                        status="blocked",
                        message=(
                            "SQL access was blocked by schema "
                            "and privacy policy."
                        ),
                        mcp_tool="execute_validated_read_query",
                    )
                    return {
                        "sql_validation_errors": [],
                        "sql_execution_error": None,
                        "security_decision": "block",
                        "security_message": (
                            "The query was blocked by database access policy."
                        ),
                        "workflow_status": "security_blocked",
                        "current_node": node_name,
                        "execution_trace": trace,
                    }

                raise RuntimeError(
                    "Agent 3b rejected the SQL with "
                    "a non-retryable error."
                )

            query_payload = tool_result.get("query_result")
            if not isinstance(query_payload, dict):
                raise RuntimeError(
                    "The validated-query MCP tool returned no query result."
                )
            query_result = QueryExecutionResult.model_validate(
                query_payload
            )

            trace = add_trace_event(
                state=state,
                node=node_name,
                status="completed",
                message=(
                    "SQL passed independent validation "
                    "and executed successfully."
                ),
                mcp_tool="execute_validated_read_query",
            )

            return {
                "query_result": (
                    query_result
                ),
                "sql_validation_errors": [],
                "sql_execution_error": None,
                "workflow_status": "analyzing",
                "current_node": node_name,
                "error_type": None,
                "error_message": None,
                "execution_trace": trace,
            }

        except Exception as exc:
            return self._error_update(
                state=state,
                node_name=node_name,
                error=exc,
                mcp_tool=(
                    "execute_validated_read_query"
                    if mcp_tool_called
                    else None
                ),
            )

    def analysis_node(
        self,
        state: GraphState,
    ) -> dict[str, Any]:
        """
        Execute Agent 4: Analysis.

        Agent 4 receives the actual query result returned by Agent 3b.

        Reads:
            user_question
            intent
            plan
            query_result

        Writes:
            analysis_result
            workflow_status
            current_node
            execution_trace
        """

        node_name = "analysis_agent"

        try:
            question = self._get_required_question(
                state
            )

            intent = self._get_required_intent(
                state
            )

            if intent.clarification_required:
                raise ValueError(
                    "Analysis cannot begin while "
                    "clarification is required."
                )

            plan = self._get_required_plan(
                state
            )

            query_result = state.get(
                "query_result"
            )

            if query_result is None:
                raise ValueError(
                    "GraphState does not contain "
                    "QueryExecutionResult."
                )

            analysis_result = (
                self.analysis_agent.analyze(
                    user_question=question,
                    intent=intent,
                    plan=plan,
                    query_result=query_result,
                )
            )

            trace = add_trace_event(
                state=state,
                node=node_name,
                status="completed",
                message=(
                    "Query results were analyzed using "
                    "deterministic statistics and "
                    "evidence-grounded interpretation."
                ),
            )

            return {
                "analysis_result": analysis_result,
                "workflow_status": "reporting",
                "current_node": node_name,
                "error_type": None,
                "error_message": None,
                "execution_trace": trace,
            }

        except Exception as exc:
            return self._error_update(
                state=state,
                node_name=node_name,
                error=exc,
            )

    def reporting_node(
        self,
        state: GraphState,
    ) -> dict[str, Any]:
        """
        Execute Agent 5: Reporting.

        Agent 5 converts the validated analysis and query result into
        a business-friendly final report.

        Reads:
            user_question
            intent
            analysis_result
            query_result

        Writes:
            final_report
            workflow_status
            current_node
            execution_trace
        """

        node_name = "reporting_agent"

        try:
            question = self._get_required_question(
                state
            )

            intent = self._get_required_intent(
                state
            )

            if intent.clarification_required:
                raise ValueError(
                    "Reporting cannot begin while "
                    "clarification is required."
                )

            analysis_result = state.get(
                "analysis_result"
            )

            if analysis_result is None:
                raise ValueError(
                    "GraphState does not contain "
                    "AnalysisResult."
                )

            query_result = state.get(
                "query_result"
            )

            if query_result is None:
                raise ValueError(
                    "GraphState does not contain "
                    "QueryExecutionResult."
                )

            regeneration_count = state.get(
                "report_regeneration_count",
                0,
            )

            if state.get("final_report") is not None:
                regeneration_count += 1

            if regeneration_count > state.get(
                "max_report_regenerations",
                0,
            ):
                raise ValueError(
                    "Report regeneration limit was exceeded."
                )

            validation_issues = state.get(
                "report_validation_feedback",
                [],
            )

            report_arguments = {
                "user_question": question,
                "intent": intent,
                "analysis_result": analysis_result,
                "query_result": query_result,
            }

            if validation_issues:
                report_arguments[
                    "validation_issues"
                ] = validation_issues

            final_report = (
                self.reporting_agent.create_report(
                    **report_arguments
                )
            )

            trace = add_trace_event(
                state=state,
                node=node_name,
                status="completed",
                message=(
                    "Business report, chart specification, "
                    "root-cause indicators, and recommendations "
                    "were generated."
                ),
            )

            return {
                "final_report": final_report,

                # Clear previous validation results whenever a new report
                # is generated.
                "deterministic_validation_result": None,
                "output_evaluation": None,
                "output_guard_result": None,
                "sanitized_output_payload": None,
                "unified_output_decision": None,
                "report_validation_feedback": [],
                "report_regeneration_count": (
                    regeneration_count
                ),
                "output_validation_message": None,

                "workflow_status": "validating_output",
                "current_node": node_name,
                "error_type": None,
                "error_message": None,
                "execution_trace": trace,
            }

        except Exception as exc:
            return self._error_update(
                state=state,
                node_name=node_name,
                error=exc,
            )

    def _build_output_technical_details(
        self,
        state: GraphState,
    ) -> dict[str, Any]:
        """
        Build technical metadata for OutputGuard filtering.

        This dictionary must not include:

        - Original user question
        - Complete database schema
        - Prompts
        - Environment variables
        - Credentials
        - Query-result rows
        """

        sql_result = state.get(
            "sql_result"
        )

        referenced_tables: list[str] = []

        if sql_result is not None:
            referenced_tables = list(
                sql_result.referenced_tables
            )

        intent = state.get("intent")
        plan = state.get("plan")

        return {
            "generated_sql": state.get(
                "generated_sql"
            ),
            "referenced_tables": (
                referenced_tables
            ),
            "sql_retry_count": state.get(
                "sql_retry_count",
                0,
            ),
            "max_sql_retries": state.get(
                "max_sql_retries",
                0,
            ),
            "intent": (
                intent.model_dump(
                    mode="json"
                )
                if intent is not None
                else None
            ),
            "plan": (
                plan.model_dump(
                    mode="json"
                )
                if plan is not None
                else None
            ),
            "execution_trace": list(
                state.get(
                    "execution_trace",
                    [],
                )
            ),
        }

    def unified_output_evaluator_node(
        self,
        state: GraphState,
    ) -> dict[str, Any]:
        """Perform all post-Agent-5 validation and protection."""
        node_name = "unified_output_evaluator"

        try:
            question = self._get_required_question(state)
            generated_sql = state.get("generated_sql")
            query_result = state.get("query_result")
            analysis_result = state.get("analysis_result")
            final_report = state.get("final_report")

            if not isinstance(generated_sql, str) or not generated_sql.strip():
                raise ValueError(
                    "Unified evaluation requires generated SQL."
                )
            if not isinstance(query_result, QueryExecutionResult):
                raise ValueError(
                    "Unified evaluation requires QueryExecutionResult."
                )
            if analysis_result is None:
                raise ValueError(
                    "Unified evaluation requires AnalysisResult."
                )
            if final_report is None:
                raise ValueError(
                    "Unified evaluation requires FinalReport."
                )

            evaluation_result = (
                self.unified_output_evaluator.evaluate(
                    user_question=question,
                    generated_sql=generated_sql,
                    query_result=query_result,
                    analysis_result=analysis_result,
                    final_report=final_report,
                    allowed_output_columns=set(query_result.columns),
                    technical_details=(
                        self._build_output_technical_details(state)
                    ),
                )
            )
            decision = evaluation_result.decision
            regeneration_count = state.get(
                "report_regeneration_count",
                0,
            )
            maximum_regenerations = state.get(
                "max_report_regenerations",
                0,
            )

            if (
                decision == "regenerate"
                and regeneration_count >= maximum_regenerations
            ):
                decision = "review"

            if decision in {"allow", "allow_redacted"}:
                workflow_status = "completed"
                trace_status = "completed"
                trace_message = (
                    "The report passed unified quality and security checks."
                    if decision == "allow"
                    else (
                        "The report passed unified checks after "
                        "deterministic redaction."
                    )
                )
                validation_message = (
                    evaluation_result.guard_result.user_message
                )
            elif decision == "block":
                workflow_status = "output_blocked"
                trace_status = "blocked"
                trace_message = (
                    "The report was blocked by output security controls."
                )
                validation_message = (
                    evaluation_result.guard_result.user_message
                )
            elif decision == "regenerate":
                workflow_status = "regenerating_report"
                trace_status = "regenerating"
                trace_message = (
                    "The report is being regenerated after validation."
                )
                validation_message = (
                    "The report needs correction before it can be returned."
                )
            else:
                workflow_status = "output_review_required"
                trace_status = "review_required"
                trace_message = (
                    "The report requires controlled quality review."
                )
                validation_message = (
                    "The report could not be approved within the "
                    "allowed regeneration attempts."
                )

            trace = add_trace_event(
                state=state,
                node=node_name,
                status=trace_status,
                message=trace_message,
            )
            guard_result = evaluation_result.guard_result

            return {
                "deterministic_validation_result": (
                    evaluation_result.deterministic_result
                ),
                "output_evaluation": (
                    evaluation_result.semantic_result
                ),
                "output_guard_result": guard_result,
                "sanitized_output_payload": (
                    guard_result.sanitized_payload
                ),
                "unified_output_decision": decision,
                "report_validation_feedback": (
                    evaluation_result.validation_feedback
                    if decision == "regenerate"
                    else []
                ),
                "output_validation_message": validation_message,
                "workflow_status": workflow_status,
                "current_node": node_name,
                "error_type": None,
                "error_message": None,
                "execution_trace": trace,
            }
        except Exception as exc:
            return self._error_update(
                state=state,
                node_name=node_name,
                error=exc,
            )




    def security_block_node(
            self,
            state: GraphState,
        ) -> dict[str, Any]:
            """
            End a prohibited request safely.

            This is a controlled security outcome, not an application error.
            Agent 1 and all database-related nodes must not run afterward.
            """

            node_name = "security_block"

            security_message = state.get(
                "security_message"
            )

            if not isinstance(
                security_message,
                str,
            ) or not security_message.strip():
                security_message = (
                    "The request cannot be processed "
                    "because it violates application "
                    "security policy."
                )

            trace = add_trace_event(
                state=state,
                node=node_name,
                status="blocked",
                message=(
                    "Workflow stopped before Agent 1 "
                    "because of a security-policy decision."
                ),
            )

            return {
                "security_decision": "block",
                "security_message": (
                    security_message
                ),
                "security_review_required": False,
                "workflow_status": (
                    "security_blocked"
                ),
                "current_node": node_name,

                # Security blocking is deliberately separate from a
                # technical workflow failure.
                "error_type": None,
                "error_message": None,

                "execution_trace": trace,
            }

    def security_review_node(
            self,
            state: GraphState,
        ) -> dict[str, Any]:
            """
            Pause an ambiguous or protected request for review.

            The current MVP stops safely rather than sending the request to
            Agent 1 or the database.
            """

            node_name = "security_review"

            security_message = state.get(
                "security_message"
            )

            if not isinstance(
                security_message,
                str,
            ) or not security_message.strip():
                security_message = (
                    "The request requires security review "
                    "before it can be processed."
                )

            trace = add_trace_event(
                state=state,
                node=node_name,
                status="review_required",
                message=(
                    "Workflow stopped before Agent 1 "
                    "because security review is required."
                ),
            )

            return {
                "security_decision": "review",
                "security_message": (
                    security_message
                ),
                "security_review_required": True,
                "workflow_status": (
                    "security_review_required"
                ),
                "current_node": node_name,
                "error_type": None,
                "error_message": None,
                "execution_trace": trace,
            }
            
    def failure_node(
        self,
        state: GraphState,
    ) -> dict[str, Any]:
        """
        End the workflow through a controlled failure route.
        """

        node_name = "failure"

        error_type = state.get(
            "error_type"
        )

        error_message = state.get(
            "error_message"
        )

        sql_retry_count = state.get(
            "sql_retry_count",
            0,
        )

        max_sql_retries = state.get(
            "max_sql_retries",
            0,
        )

        if (
            not error_message
            and isinstance(
                sql_retry_count,
                int,
            )
            and isinstance(
                max_sql_retries,
                int,
            )
            and sql_retry_count
            >= max_sql_retries
        ):
            error_type = (
                "SQLRetryLimitExceeded"
            )

            error_message = (
                "SQL validation or execution failed "
                "after the maximum number of retries."
            )

        safe_error_type = (
            error_type
            if isinstance(error_type, str)
            and error_type.strip()
            else "WorkflowError"
        )

        safe_error_message = (
            error_message
            if isinstance(error_message, str)
            and error_message.strip()
            else "The workflow could not be completed."
        )

        trace = add_trace_event(
            state=state,
            node=node_name,
            status="failed",
            message=(
                "Workflow ended through the "
                "controlled failure route."
            ),
        )

        return {
            "workflow_status": "failed",
            "current_node": node_name,
            "error_type": safe_error_type,
            "error_message": safe_error_message,
            "execution_trace": trace,
        }


def build_workflow_nodes(
    mcp_client: InternalMCPClient,
) -> WorkflowNodes:
    """
    Build all current workflow nodes.

    Ollama model allocation:

    - llama3.2:latest:
      Intent Agent and Planning Agent

    - sqlcoder:latest:
      SQL Generation Agent

    Database access remains read-only.
    """

    input_guard = InputGuard()

    prompt_injection_guard = (
        PromptInjectionGuard(
            model="llama3.2:latest",
            host="http://localhost:11434",
            context_length=4096,
            max_output_tokens=700,
            maximum_attempts=2,
            blocking_confidence_threshold=0.80,
            review_confidence_threshold=0.60,
        )
    )


    intent_agent = IntentAgent(
        model="llama3.2:latest",
        host="http://localhost:11434",
    )

    planning_agent = PlanningAgent(
        model="llama3.2:latest",
        host="http://localhost:11434",
    )

    sql_agent = SQLAgent(
        model="sqlcoder:latest",
        host="http://localhost:11434",
        dialect="sqlite",
        context_length=8192,
        max_output_tokens=1024,
        max_attempts=3,
    )

    analysis_agent = AnalysisAgent(
    model="llama3.2:latest",
    host="http://localhost:11434",
    context_length=8192,
    max_output_tokens=1500,
    maximum_sample_rows=50,
)

    reporting_agent = ReportingAgent(
    model="llama3.2:latest",
    host="http://localhost:11434",
    context_length=8192,
    max_output_tokens=1800,
    maximum_sample_rows=30,
)

    deterministic_validator = (
        DeterministicOutputValidator(
            numeric_tolerance=0.01,
            percentage_tolerance=0.10,
            maximum_claims_to_validate=100,
        )
    )

    llm_output_evaluator = (
        LLMOutputEvaluator(
            model="llama3.2:latest",
            host="http://localhost:11434",
            context_length=8192,
            max_output_tokens=1200,
            maximum_attempts=2,
            maximum_sample_rows=30,
            approval_threshold=0.90,
            regeneration_threshold=0.80,
        )
    )

    output_guard = OutputGuard(
        maximum_output_rows=1000
    )
    unified_output_evaluator = UnifiedOutputEvaluator(
        deterministic_checks=deterministic_validator,
        semantic_judge=llm_output_evaluator,
        output_sanitizer=output_guard,
        role=UserRole.ANALYST,
    )
    return WorkflowNodes(
    input_guard=input_guard,
    prompt_injection_guard=(
        prompt_injection_guard
    ),
    intent_agent=intent_agent,
    planning_agent=planning_agent,
    sql_agent=sql_agent,
    analysis_agent=analysis_agent,
    reporting_agent=reporting_agent,
    unified_output_evaluator=(
        unified_output_evaluator
    ),
    mcp_client=mcp_client,
)
