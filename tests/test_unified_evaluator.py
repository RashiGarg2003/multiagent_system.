from __future__ import annotations

from typing import Any

from evaluation.unified_evaluator import UnifiedOutputEvaluator
from models.schemas import (
    AnalysisResult,
    FinalReport,
    QueryExecutionResult,
)
from security.models import (
    DeterministicValidationResult,
    EvaluationDecision,
    OutputEvaluation,
    SecurityDecision,
)
from security.output_guard import OutputGuard
from workflow.graph import build_graph
from workflow.nodes import WorkflowNodes
from workflow.routing import route_after_unified_output_evaluator


class StubDeterministicChecks:
    def __init__(
        self,
        result: DeterministicValidationResult,
    ) -> None:
        self.result = result

    def validate(self, **_: Any) -> DeterministicValidationResult:
        return self.result


class StubSemanticJudge:
    def __init__(
        self,
        result: OutputEvaluation,
    ) -> None:
        self.result = result
        self.calls = 0

    def evaluate(self, **_: Any) -> OutputEvaluation:
        self.calls += 1
        return self.result


def _validation_result(
    decision: EvaluationDecision = EvaluationDecision.APPROVE,
) -> DeterministicValidationResult:
    is_approved = decision == EvaluationDecision.APPROVE
    return DeterministicValidationResult(
        valid=is_approved,
        schema_valid=is_approved,
        sql_safe=is_approved,
        numerical_claims_valid=is_approved,
        ranking_valid=is_approved,
        chart_valid=is_approved,
        pii_safe=is_approved,
        secret_safe=is_approved,
        prompt_leakage_safe=is_approved,
        causal_language_safe=is_approved,
        numeric_claims=[],
        verified_claims=[],
        unsupported_claims=[],
        privacy_findings=[],
        issues=[] if is_approved else ["Unsafe output."],
        decision=decision,
    )


def _semantic_result(
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
        relevance_score=0.95,
        groundedness_score=0.95,
        accuracy_score=0.95,
        safety_score=1.0,
        overall_score=0.95,
        issues=[],
        decision=decision,
        explanation="The report is relevant and evidence-grounded.",
    )


def _inputs(
    summary: str = "Sales increased in the West region.",
) -> tuple[QueryExecutionResult, AnalysisResult, FinalReport]:
    query_result = QueryExecutionResult(
        columns=["region", "sales"],
        rows=[{"region": "West", "sales": 10}],
        row_count=1,
        execution_time_ms=1.0,
        truncated=False,
        query="SELECT region, sales FROM sales",
    )
    analysis_result = AnalysisResult(
        summary="West region sales were 10.",
        findings=[],
        trends=[],
        anomalies=[],
        possible_drivers=[],
        limitations=[],
    )
    final_report = FinalReport(
        business_summary=summary,
        key_findings=[],
        root_cause_indicators=[],
        recommendations=[],
        chart=None,
        limitations=[],
    )
    return query_result, analysis_result, final_report


def _evaluate(
    deterministic: DeterministicValidationResult,
    semantic: StubSemanticJudge,
    final_report: FinalReport,
) -> Any:
    query_result, analysis_result, _ = _inputs()
    evaluator = UnifiedOutputEvaluator(
        deterministic_checks=StubDeterministicChecks(deterministic),
        semantic_judge=semantic,
        output_sanitizer=OutputGuard(),
    )
    return evaluator.evaluate(
        user_question="Show sales by region.",
        generated_sql=query_result.query,
        query_result=query_result,
        analysis_result=analysis_result,
        final_report=final_report,
        allowed_output_columns=set(query_result.columns),
        technical_details={},
    )


def test_unified_evaluator_redacts_pii_before_allowing_output() -> None:
    _, _, report = _inputs(
        "Contact person@example.com about the West region."
    )
    semantic = StubSemanticJudge(_semantic_result())

    result = _evaluate(
        _validation_result(),
        semantic,
        report,
    )

    assert result.decision == "allow_redacted"
    assert result.guard_result.decision == SecurityDecision.ALLOW_REDACTED
    assert "person@example.com" not in (
        result.guard_result.sanitized_payload["final_report"][
            "business_summary"
        ]
    )
    assert semantic.calls == 1


def test_security_block_skips_semantic_judgment() -> None:
    _, _, report = _inputs()
    semantic = StubSemanticJudge(_semantic_result())

    result = _evaluate(
        _validation_result(EvaluationDecision.BLOCK),
        semantic,
        report,
    )

    assert result.decision == "block"
    assert result.guard_result.decision == SecurityDecision.BLOCK
    assert semantic.calls == 0


def test_report_regeneration_routes_only_below_limit() -> None:
    state = {
        "workflow_status": "regenerating_report",
        "unified_output_decision": "regenerate",
        "report_regeneration_count": 1,
        "max_report_regenerations": 2,
    }

    assert (
        route_after_unified_output_evaluator(state)
        == "regenerate_report"
    )

    exhausted_state = {
        **state,
        "workflow_status": "output_review_required",
        "unified_output_decision": "review",
        "report_regeneration_count": 2,
    }
    assert (
        route_after_unified_output_evaluator(exhausted_state)
        == "output_review"
    )


def test_compiled_graph_registers_only_unified_output_evaluator() -> None:
    workflow_nodes = object.__new__(WorkflowNodes)
    node_names = (
        "input_guard_node",
        "prompt_injection_guard_node",
        "security_block_node",
        "security_review_node",
        "intent_node",
        "clarification_node",
        "planning_node",
        "schema_node",
        "sql_generation_node",
        "sql_validation_execution_node",
        "analysis_node",
        "reporting_node",
        "unified_output_evaluator_node",
        "failure_node",
    )
    for node_name in node_names:
        setattr(workflow_nodes, node_name, lambda _: {})

    graph = build_graph(workflow_nodes)
    compiled = graph.get_graph()
    evaluator_nodes = [
        node_id
        for node_id in compiled.nodes
        if "evaluator" in node_id or "guard" in node_id
    ]

    assert evaluator_nodes == [
        "input_guard",
        "prompt_injection_guard",
        "unified_output_evaluator",
    ]
    assert any(
        edge.source == "reporting_agent"
        and edge.target == "unified_output_evaluator"
        for edge in compiled.edges
    )
