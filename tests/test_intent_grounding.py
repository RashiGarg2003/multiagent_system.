"""Tests that extracted constraints are grounded in the user question."""

from agents.intent_agent import IntentAgent
from models.schemas import (
    FilterCondition,
    IntentResult,
    TimePeriod,
)


def build_intent(
    *,
    filters: list[FilterCondition] | None = None,
    time_period: TimePeriod | None = None,
    limit: int | None = None,
) -> IntentResult:
    return IntentResult(
        original_question="Show the top 3 products by revenue.",
        rewritten_question="Rank products by revenue.",
        analysis_type="ranking",
        metrics=["revenue"],
        dimensions=["product"],
        filters=filters or [],
        time_period=time_period or TimePeriod(),
        sort_order="descending",
        limit=limit,
    )


def test_removes_ungrounded_filters_and_period_and_recovers_top_limit():
    intent = build_intent(
        filters=[
            FilterCondition(
                field="product",
                operator="eq",
                value="all",
            )
        ],
        time_period=TimePeriod(
            start_date="2020-01-01",
            end_date="2022-12-31",
        ),
    )

    grounded = IntentAgent()._ground_explicit_constraints(
        result=intent,
        user_question="Show the top 3 products by revenue.",
    )

    assert grounded.filters == []
    assert grounded.time_period == TimePeriod()
    assert grounded.limit == 3


def test_preserves_filter_and_period_explicitly_named_in_question():
    intent = build_intent(
        filters=[
            FilterCondition(
                field="category",
                operator="equals",
                value="Beverages",
            )
        ],
        time_period=TimePeriod(
            start_date="2025-01-01",
            end_date="2025-12-31",
        ),
    )

    grounded = IntentAgent()._ground_explicit_constraints(
        result=intent,
        user_question=(
            "Show the top 3 products by revenue in 2025 "
            "where category is Beverages."
        ),
    )

    assert grounded.filters == intent.filters
    assert grounded.time_period == intent.time_period
    assert grounded.limit == 3


def test_monthly_request_keeps_granularity_but_drops_invented_dates():
    intent = build_intent(
        time_period=TimePeriod(
            start_date="2020-01-01",
            end_date="2022-12-31",
            current_period="monthly",
            granularity="month",
        )
    )

    grounded = IntentAgent()._ground_explicit_constraints(
        result=intent,
        user_question="Show monthly revenue trend.",
    )

    assert grounded.time_period == TimePeriod(
        granularity="month"
    )
