Multi-Agent Data Analytics System
=================================

FastAPI + Streamlit + LangGraph + Ollama + MCP implementation.

## MCP server

The project exposes Model Context Protocol tools, resources, and prompts.

### Stdio transport

```powershell
python mcp_server.py
```

### Streamable HTTP transport

Start FastAPI on port 8000:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

The MCP endpoint is mounted at:

```text
http://127.0.0.1:8000/mcp/
```

### MCP resources

- `northwind://schema` - read-only database tables, views, and columns
- `analytics://models` - Ollama availability and required model status

### MCP tools

- `list_database_objects`
- `get_table_columns`
- `execute_read_sql`
- `run_analytics`

### MCP prompt

- `analyze_northwind`