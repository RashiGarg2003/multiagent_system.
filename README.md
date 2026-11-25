# Multi-Agent Data Analytics System

FastAPI + Streamlit + LangGraph + local Ollama models + SQLite.

## Architecture

- LangGraph calls one lifecycle-managed stdio MCP server through a reusable
  `ClientSession`.
- The internal MCP surface contains only `get_approved_schema` and
  `execute_validated_read_query`; database access remains read-only and
  policy-validated.
- Agent 5 is followed by one `UnifiedOutputEvaluator`. It combines deterministic
  checks, semantic quality judgment, security filtering, redaction, and the
  final sanitized response payload.
- Reports can be regenerated at most twice. Security violations block without
  regeneration.
- Clarification pauses are checkpointed. The API returns a thread ID, screens
  the answer, and resumes the same LangGraph thread at Agent 1.
- FastAPI returns additive safe execution-trace metadata. Streamlit displays
  it in a collapsed expander and renders charts locally with `ChartTool`.

## Run locally

Start FastAPI:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Start Streamlit in another terminal:

```powershell
streamlit run app.py
```

## Internal MCP server

The stdio server is managed by FastAPI for the application lifetime. To run it
standalone for development:

```powershell
python mcp_server.py
```

It exposes only the two internal tools described above. No HTTP MCP endpoint
is mounted by FastAPI.

## Focused tests

```powershell
python -m pytest tests\test_unified_evaluator.py tests\test_clarification_checkpoint.py tests\test_safe_trace.py tests\test_chart_presentation.py tests\test_api.py tests\test_mcp_server.py tests\test_sql_intent_alignment.py tests\test_schema_policy.py -q
```
