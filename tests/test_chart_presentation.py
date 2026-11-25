from models.schemas import ChartSpecification, QueryExecutionResult
from tools.chart_tool import ChartTool


def _long_category_result() -> QueryExecutionResult:
    rows = [
        {
            "Product Category": (
                f"Long product category label {index}"
            ),
            "Revenue": float(index * 1250),
        }
        for index in range(1, 14)
    ]
    return QueryExecutionResult(
        columns=["Product Category", "Revenue"],
        rows=rows,
        row_count=len(rows),
        execution_time_ms=1.0,
        truncated=False,
        query="SELECT category, revenue FROM report",
    )


def test_long_bar_categories_use_responsive_horizontal_chart() -> None:
    chart = ChartSpecification(
        chart_type="bar",
        title="Revenue by product category",
        x_column="Product Category",
        y_columns=["Revenue"],
        rationale="Compare category revenue.",
    )

    result = ChartTool().render(
        chart_specification=chart,
        query_result=_long_category_result(),
    )

    assert result.rendered_chart_type == "horizontal_bar"
    assert any("horizontal bar chart" in item for item in result.warnings)
    assert result.figure is not None
    assert result.figure.layout.height >= 360
    assert result.figure.layout.yaxis.title.text == "Product Category"
    assert result.figure.layout.xaxis.tickformat == ",.2f"
    assert "%{x:,.2f}" in result.figure.data[0].hovertemplate


def test_empty_query_results_do_not_create_a_chart() -> None:
    query_result = QueryExecutionResult(
        columns=["Category", "Revenue"],
        rows=[],
        row_count=0,
        execution_time_ms=0.0,
        truncated=False,
        query="SELECT category, revenue FROM report WHERE 1 = 0",
    )
    chart = ChartSpecification(
        chart_type="bar",
        title="Revenue",
        x_column="Category",
        y_columns=["Revenue"],
        rationale="Compare revenue.",
    )

    result = ChartTool().render(chart, query_result)

    assert result.figure is None
    assert result.used_fallback
    assert result.rendered_chart_type == "none"
