"""Lifecycle-managed synchronous bridge to the internal stdio MCP server."""

import asyncio
import json
import os
import sys
import threading
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import (
    StdioServerParameters,
    stdio_client,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_TOOLS = {
    "get_approved_schema",
    "execute_validated_read_query",
}


class MCPClientError(RuntimeError):
    """Raised when the internal MCP server cannot safely serve a tool call."""


class InternalMCPClient:
    """Own one stdio MCP process and reusable ClientSession."""

    def __init__(
        self,
        database_path: Path | str | None = None,
        maximum_result_rows: int = 1000,
        startup_timeout_seconds: float = 30.0,
    ) -> None:
        self.database_path = (
            Path(database_path).resolve()
            if database_path is not None
            else PROJECT_ROOT / "database" / "northwind.db"
        )
        self.maximum_result_rows = maximum_result_rows
        self.startup_timeout_seconds = startup_timeout_seconds
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._close_event: asyncio.Event | None = None
        self._owner_future: Any = None
        self._startup_error: BaseException | None = None
        self._tool_names: frozenset[str] = frozenset()
        self._tool_input_schemas: dict[str, dict[str, Any]] = {}
        self._lifecycle_lock = threading.Lock()

    @property
    def tool_names(self) -> frozenset[str]:
        """Return the tools advertised by the active MCP server."""
        return self._tool_names

    @property
    def tool_input_schemas(self) -> dict[str, dict[str, Any]]:
        """Return the input schemas advertised by the active MCP server."""
        return {
            name: dict(schema)
            for name, schema in self._tool_input_schemas.items()
        }

    def start(self) -> None:
        """Start the single stdio server process and initialize its session."""
        with self._lifecycle_lock:
            if self._session is not None:
                return

            self._ready.clear()
            self._thread = threading.Thread(
                target=self._run_event_loop,
                name="internal-mcp-client",
                daemon=True,
            )
            self._thread.start()

        if not self._ready.wait(self.startup_timeout_seconds):
            raise MCPClientError(
                "The internal MCP client did not start in time."
            )

        self._ready.clear()
        self._owner_future = self._submit(
            self._own_session()
        )
        if not self._ready.wait(self.startup_timeout_seconds):
            self.close()
            raise MCPClientError(
                "The internal MCP session did not initialize in time."
            )
        if self._startup_error is not None:
            error = self._startup_error
            self.close()
            raise MCPClientError(
                "The internal MCP session could not be initialized."
            ) from error

    def _run_event_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        loop.run_forever()
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

    async def _own_session(self) -> None:
        environment = {
            key: os.environ[key]
            for key in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP")
            if key in os.environ
        }
        environment.update(
            {
                "PYTHONPATH": str(PROJECT_ROOT),
                "DATABASE_PATH": str(self.database_path),
                "MAXIMUM_RESULT_ROWS": str(
                    self.maximum_result_rows
                ),
            }
        )
        server_parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_layer.server"],
            env=environment,
            cwd=PROJECT_ROOT,
        )

        self._stack = AsyncExitStack()
        try:
            read_stream, write_stream = (
                await self._stack.enter_async_context(
                    stdio_client(server_parameters)
                )
            )
            self._session = await self._stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self._session.initialize()
            result = await self._session.list_tools()
            available_tools = {tool.name for tool in result.tools}
            if available_tools != EXPECTED_TOOLS:
                raise MCPClientError(
                    "The internal MCP server exposed an unexpected tool set."
                )
            self._tool_names = frozenset(available_tools)
            self._tool_input_schemas = {
                tool.name: dict(tool.inputSchema)
                for tool in result.tools
            }
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            await self._stack.aclose()
            self._session = None
            self._stack = None
            return

        self._close_event = asyncio.Event()
        self._ready.set()
        await self._close_event.wait()
        await self._stack.aclose()
        self._session = None
        self._stack = None

    def _submit(self, coroutine: Any):
        loop = self._loop
        if loop is None or not loop.is_running():
            coroutine.close()
            raise MCPClientError(
                "The internal MCP event loop is not running."
            )
        return asyncio.run_coroutine_threadsafe(coroutine, loop)

    async def _call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        session = self._session
        if session is None:
            raise MCPClientError(
                "The internal MCP session is not available."
            )
        return await session.call_tool(name, arguments)

    @staticmethod
    def _decode_result(result: Any) -> dict[str, Any]:
        if getattr(result, "isError", False) or getattr(
            result, "is_error", False
        ):
            raise MCPClientError(
                "The internal MCP tool returned an error."
            )

        structured_content = getattr(
            result,
            "structuredContent",
            None,
        )
        if structured_content is None:
            structured_content = getattr(
                result,
                "structured_content",
                None,
            )
        if isinstance(structured_content, dict):
            return structured_content

        for content in getattr(result, "content", []):
            text = getattr(content, "text", None)
            if isinstance(text, str):
                try:
                    decoded = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(decoded, dict):
                    return decoded

        raise MCPClientError(
            "The internal MCP tool returned an invalid response."
        )

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        timeout_seconds: float = 120.0,
    ) -> dict[str, Any]:
        """Call an internal MCP tool through the persistent ClientSession."""
        if name not in EXPECTED_TOOLS:
            raise ValueError("Unsupported internal MCP tool.")
        if self._session is None:
            raise MCPClientError(
                "The internal MCP session is not available."
            )

        try:
            result = self._submit(
                self._call_tool(name, arguments)
            ).result(timeout=timeout_seconds)
            return self._decode_result(result)
        except MCPClientError:
            raise
        except Exception as exc:
            raise MCPClientError(
                "The internal MCP tool call failed."
            ) from exc

    def close(self) -> None:
        """Close the session and stop the client event loop."""
        loop = self._loop
        thread = self._thread
        if loop is None or thread is None:
            return

        if loop.is_running():
            try:
                close_event = self._close_event
                if close_event is not None:
                    loop.call_soon_threadsafe(close_event.set)
                if self._owner_future is not None:
                    self._owner_future.result(
                        timeout=self.startup_timeout_seconds
                    )
            finally:
                loop.call_soon_threadsafe(loop.stop)
                thread.join(timeout=self.startup_timeout_seconds)

        self._loop = None
        self._thread = None
        self._owner_future = None
        self._close_event = None
        self._tool_names = frozenset()
        self._tool_input_schemas = {}
        self._ready.clear()

    def __enter__(self) -> "InternalMCPClient":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
