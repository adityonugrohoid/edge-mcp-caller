#!/usr/bin/env python3
"""MCP Client Bridge — translates model JSON output to MCP tools/call protocol.

Takes the specialist model's JSON output:
    {"tool": "list_directory", "args": {"path": "src/"}}

And executes it against a real MCP filesystem server via the standard protocol.
"""

import json
import logging
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
import mcp.types as types

logger = logging.getLogger(__name__)

VALID_TOOLS = {"list_directory", "read_file", "search_files", "write_file", "create_directory"}


def parse_model_output(raw: str) -> dict:
    """Parse the specialist model's JSON string into tool + args.

    Args:
        raw: Raw string from model's message.content field.

    Returns:
        {"tool": str | None, "args": dict, "error": str | None}
    """
    result: dict = {"tool": None, "args": {}, "error": None}

    text = raw.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines[1:] if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        result["error"] = f"invalid JSON: {e}"
        return result

    tool = parsed.get("tool")
    if not tool:
        result["error"] = "missing 'tool' field in model output"
        return result

    if tool not in VALID_TOOLS:
        result["error"] = f"unknown tool '{tool}' (valid: {', '.join(sorted(VALID_TOOLS))})"
        return result

    result["tool"] = tool
    result["args"] = parsed.get("args", {})
    return result


class MCPClientBridge:
    """Bridge between model JSON output and a live MCP filesystem server.

    Usage:
        async with MCPClientBridge(["/home/user/project"]) as bridge:
            tools = await bridge.list_tools()
            result = await bridge.call_tool("list_directory", {"path": "."})
            print(result)
    """

    def __init__(self, allowed_dirs: list[str]) -> None:
        self._allowed_dirs = [str(Path(d).resolve()) for d in allowed_dirs]
        self._exit_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def connect(self) -> None:
        """Start the MCP filesystem server and initialize the client session."""
        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem"] + self._allowed_dirs,
        )

        self._exit_stack = AsyncExitStack()
        read, write = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()
        logger.info("MCP session initialized (allowed: %s)", self._allowed_dirs)

    async def list_tools(self) -> list[dict]:
        """List tools available on the MCP server.

        Returns:
            List of {"name": str, "description": str} dicts.
        """
        self._ensure_connected()
        result = await self._session.list_tools()
        return [
            {"name": t.name, "description": t.description or ""}
            for t in result.tools
        ]

    async def call_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool call on the MCP server.

        Args:
            tool_name: One of the valid MCP filesystem tools.
            args: Arguments dict matching the tool's input schema.

        Returns:
            Text result from the MCP server.

        Raises:
            ValueError: If tool_name is not in VALID_TOOLS.
            RuntimeError: If not connected.
        """
        self._ensure_connected()

        if tool_name not in VALID_TOOLS:
            raise ValueError(
                f"Unknown tool '{tool_name}' (valid: {', '.join(sorted(VALID_TOOLS))})"
            )

        result = await self._session.call_tool(tool_name, arguments=args)

        # Extract text from result content blocks
        parts = []
        for block in result.content:
            if isinstance(block, types.TextContent):
                parts.append(block.text)
            else:
                parts.append(str(block))

        return "\n".join(parts)

    async def execute_model_output(self, raw: str) -> dict:
        """Parse model output and execute the tool call end-to-end.

        Args:
            raw: Raw JSON string from the model's response.

        Returns:
            {"tool": str, "args": dict, "result": str | None, "error": str | None}
        """
        parsed = parse_model_output(raw)
        response: dict = {
            "tool": parsed["tool"],
            "args": parsed["args"],
            "result": None,
            "error": parsed["error"],
        }

        if parsed["error"]:
            return response

        try:
            response["result"] = await self.call_tool(parsed["tool"], parsed["args"])
        except Exception as e:
            response["error"] = f"MCP call failed: {e}"

        return response

    async def close(self) -> None:
        """Shut down the MCP server connection."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None
            logger.info("MCP session closed")

    async def __aenter__(self) -> "MCPClientBridge":
        await self.connect()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    def _ensure_connected(self) -> None:
        if self._session is None:
            raise RuntimeError("Not connected — call connect() or use async with")
