from workflow.graph import _with_safe_trace_timing
from workflow.state import add_trace_event, create_initial_state


def test_timed_node_adds_only_safe_trace_metadata() -> None:
    state = create_initial_state("Show sales by region.")
    state["sql_retry_count"] = 1
    state["report_regeneration_count"] = 0

    def node(current_state):
        return {
            "execution_trace": add_trace_event(
                state=current_state,
                node="sql_validation_execution",
                status="completed",
                message="SQL passed independent validation.",
                mcp_tool="execute_validated_read_query",
            )
        }

    update = _with_safe_trace_timing(
        "sql_validation_execution",
        node,
    )(state)
    event = update["execution_trace"][-1]

    assert event["stage"] == "sql_validation_execution"
    assert event["duration_ms"] >= 0
    assert event["mcp_tool"] == "execute_validated_read_query"
    assert event["sql_retry_count"] == 1
    assert event["report_retry_count"] == 0
    assert "message" in event
