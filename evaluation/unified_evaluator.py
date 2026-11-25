"""Single post-Agent-5 evaluator combining checks, semantics, and protection."""

from dataclasses import dataclass
from typing import Any, Literal

from models.schemas import (
    AnalysisResult,
    FinalReport,
    QueryExecutionResult,
)
from security.models import (
    DeterministicValidationResult,
    EvaluationDecision,
    OutputEvaluation,
    OutputGuardResult,
    UserRole,
)


UnifiedDecision = Literal[
    "allow",
    "allow_redacted",
    "regenerate",
    "review",
    "block",
]


@dataclass(frozen=True)
class UnifiedEvaluationResult:
    """Private result bundle for the one evaluator workflow node."""

    decision: UnifiedDecision
    deterministic_result: DeterministicValidationResult
    semantic_result: OutputEvaluation
    guard_result: OutputGuardResult
    validation_feedback: list[str]


class UnifiedOutputEvaluator:
    """Run deterministic, semantic, and security checks as one decision."""

    def __init__(
        self,
        deterministic_checks: Any,
        semantic_judge: Any,
        output_sanitizer: Any,
        role: UserRole = UserRole.ANALYST,
    ) -> None:
        self.deterministic_checks = deterministic_checks
        self.semantic_judge = semantic_judge
        self.output_sanitizer = output_sanitizer
        self.role = UserRole(role)

    @staticmethod
    def _placeholder_evaluation(
        decision: EvaluationDecision = EvaluationDecision.APPROVE,
    ) -> OutputEvaluation:
        return OutputEvaluation(
            answers_user_question=True,
            grounded_in_query_result=True,
            numerical_claims_supported=True,
            findings_supported=True,
            recommendations_supported=True,
            root_cause_language_cautious=True,
            chart_matches_result=True,
            contains_sensitive_data=False,
            contains_prompt_leakage=False,
            contains_secret_data=False,
            relevance_score=1.0,
            groundedness_score=1.0,
            accuracy_score=1.0,
            safety_score=1.0,
            overall_score=1.0,
            issues=[],
            decision=decision,
            explanation="Security enforcement is deterministic.",
        )

    @staticmethod
    def _enum_value(value: Any) -> str | None:
        primitive = getattr(value, "value", value)
        return primitive if isinstance(primitive, str) else None

    @staticmethod
    def _semantic_feedback(
        evaluation: OutputEvaluation,
    ) -> list[str]:
        feedback: list[str] = []
        if not evaluation.answers_user_question:
            feedback.append(
                "Answer the analytical question more directly."
            )
        if not evaluation.grounded_in_query_result:
            feedback.append(
                "Ground the summary and findings in the query results."
            )
        if not evaluation.findings_supported:
            feedback.append(
                "Use only findings supported by the analysis evidence."
            )
        if not evaluation.recommendations_supported:
            feedback.append(
                "Tie recommendations to supported findings."
            )
        if not evaluation.root_cause_language_cautious:
            feedback.append(
                "Describe possible drivers cautiously; do not state causation as proven."
            )
        if not evaluation.chart_matches_result:
            feedback.append(
                "Choose a chart specification that matches the returned columns and values."
            )
        return feedback

    def evaluate(
        self,
        *,
        user_question: str,
        generated_sql: str,
        query_result: QueryExecutionResult,
        analysis_result: AnalysisResult,
        final_report: FinalReport,
        allowed_output_columns: set[str],
        technical_details: dict[str, Any],
    ) -> UnifiedEvaluationResult:
        deterministic = self.deterministic_checks.validate(
            user_question=user_question,
            generated_sql=generated_sql,
            query_result=query_result,
            analysis_result=analysis_result,
            final_report=final_report,
        )
        deterministic_decision = self._enum_value(
            deterministic.decision
        )

        placeholder = self._placeholder_evaluation(
            EvaluationDecision.BLOCK
            if deterministic_decision == "block"
            else EvaluationDecision.APPROVE
        )
        preflight_guard = self.output_sanitizer.guard(
            final_report=final_report,
            query_result=query_result,
            deterministic_result=deterministic,
            output_evaluation=placeholder,
            role=self.role,
            allowed_output_columns=allowed_output_columns,
            technical_details=technical_details,
        )

        if (
            deterministic_decision == "block"
            or self._enum_value(preflight_guard.decision) == "block"
        ):
            semantic = self._placeholder_evaluation(
                EvaluationDecision.BLOCK
            )
            return UnifiedEvaluationResult(
                decision="block",
                deterministic_result=deterministic,
                semantic_result=semantic,
                guard_result=preflight_guard,
                validation_feedback=[],
            )

        safe_payload = preflight_guard.sanitized_payload
        if not isinstance(safe_payload, dict):
            raise RuntimeError(
                "Output sanitization did not return a payload."
            )

        safe_report = FinalReport.model_validate(
            safe_payload["final_report"]
        )
        safe_query_payload = safe_payload["query_result"]
        safe_query_result = query_result.model_copy(
            update={
                "columns": safe_query_payload["columns"],
                "rows": safe_query_payload["rows"],
            }
        )
        (
            safe_analysis_payload,
            analysis_has_secret,
            analysis_has_restricted_data,
            analysis_has_prompt_leakage,
        ) = self.output_sanitizer.sanitize_internal_value(
            analysis_result.model_dump(mode="json"),
            "analysis_result",
        )
        (
            safe_sql,
            sql_has_secret,
            sql_has_restricted_data,
            sql_has_prompt_leakage,
        ) = self.output_sanitizer.sanitize_internal_value(
            generated_sql,
            "generated_sql",
        )

        if any(
            (
                analysis_has_secret,
                analysis_has_restricted_data,
                analysis_has_prompt_leakage,
                sql_has_secret,
                sql_has_restricted_data,
                sql_has_prompt_leakage,
            )
        ):
            semantic = self._placeholder_evaluation(
                EvaluationDecision.BLOCK
            )
            blocked_guard = self.output_sanitizer.guard(
                final_report=final_report,
                query_result=query_result,
                deterministic_result=deterministic,
                output_evaluation=semantic,
                role=self.role,
                allowed_output_columns=allowed_output_columns,
                technical_details=technical_details,
            )
            return UnifiedEvaluationResult(
                decision="block",
                deterministic_result=deterministic,
                semantic_result=semantic,
                guard_result=blocked_guard,
                validation_feedback=[],
            )

        safe_analysis_result = AnalysisResult.model_validate(
            safe_analysis_payload
        )

        if deterministic_decision == "regenerate":
            semantic = placeholder
        else:
            semantic = self.semantic_judge.evaluate(
                user_question=user_question,
                generated_sql=safe_sql,
                query_result=safe_query_result,
                analysis_result=safe_analysis_result,
                final_report=safe_report,
                deterministic_result=deterministic,
            )

        guard_result = self.output_sanitizer.guard(
            final_report=final_report,
            query_result=query_result,
            deterministic_result=deterministic,
            output_evaluation=semantic,
            role=self.role,
            allowed_output_columns=allowed_output_columns,
            technical_details=technical_details,
        )

        guard_decision = self._enum_value(guard_result.decision)
        semantic_decision = self._enum_value(semantic.decision)

        if guard_decision == "block":
            decision: UnifiedDecision = "block"
        elif deterministic_decision == "regenerate" or semantic_decision == "regenerate":
            decision = "regenerate"
        elif semantic_decision == "review" or guard_decision == "review":
            decision = "review"
        elif guard_decision == "allow_redacted":
            decision = "allow_redacted"
        elif guard_decision == "allow":
            decision = "allow"
        else:
            decision = "review"

        feedback = (
            list(deterministic.issues)
            if decision == "regenerate"
            else []
        )
        if decision == "regenerate" and semantic_decision == "regenerate":
            feedback.extend(self._semantic_feedback(semantic))

        return UnifiedEvaluationResult(
            decision=decision,
            deterministic_result=deterministic,
            semantic_result=semantic,
            guard_result=guard_result,
            validation_feedback=list(dict.fromkeys(feedback)),
        )
