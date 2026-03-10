#!/usr/bin/env python3
"""MCP Client Bridge — translates model JSON output to MCP tools/call protocol.

Supports 14 tools across 2 MCP servers (filesystem + git):

    {"tool": "list_directory", "args": {"path": "src/"}}     → filesystem server
    {"tool": "git_status", "args": {}}                        → git server

Arg translations handled by the bridge:
    edit_file:  {path, old_text, new_text} → MCP {path, edits: [{oldText, newText}]}
    git tools:  repo_path injected from config (model doesn't output it)
"""

import json
import logging
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
import mcp.types as types

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool → server routing
# ---------------------------------------------------------------------------

FILESYSTEM_TOOLS = {
    "list_directory", "read_file", "search_files", "write_file",
    "create_directory", "edit_file", "move_file", "directory_tree",
}
GIT_TOOLS = {
    "git_status", "git_diff_staged", "git_commit",
    "git_log", "git_branch", "git_create_branch",
}
VALID_TOOLS = FILESYSTEM_TOOLS | GIT_TOOLS

# Model tool name → MCP server tool name (identity by default).
# Add entries here if the MCP server uses different names.
_TOOL_NAME_MAP: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Arg translation
# ---------------------------------------------------------------------------


def _translate_args(tool_name: str, args: dict, repo_path: str | None) -> dict:
    """Translate model output args to MCP server format.

    - edit_file: {path, old_text, new_text} → {path, edits: [{oldText, newText}]}
    - git tools: inject repo_path from config
    - git_branch: inject branch_type="local" (server requires it, model doesn't output it)
    """
    if tool_name == "edit_file":
        return {
            "path": args.get("path", ""),
            "edits": [{
                "oldText": args.get("old_text", ""),
                "newText": args.get("new_text", ""),
            }],
        }

    if tool_name in GIT_TOOLS and repo_path:
        translated = dict(args)
        translated["repo_path"] = repo_path
        # git_branch requires branch_type; model doesn't output it
        if tool_name == "git_branch" and "branch_type" not in translated:
            translated["branch_type"] = "local"
        return translated

    return args


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


class MCPClientBridge:
    """Bridge between model JSON output and live MCP servers (filesystem + git).

    Usage:
        async with MCPClientBridge(
            allowed_dirs=["/home/user/project"],
            repo_path="/home/user/project",
        ) as bridge:
            tools = await bridge.list_tools()
            result = await bridge.call_tool("list_directory", {"path": "."})
            result = await bridge.call_tool("git_status", {})
    """

    def __init__(
        self,
        allowed_dirs: list[str] | None = None,
        repo_path: str | None = None,
    ) -> None:
        self._allowed_dirs = [str(Path(d).resolve()) for d in (allowed_dirs or [])]
        self._repo_path = str(Path(repo_path).resolve()) if repo_path else None
        self._exit_stack: AsyncExitStack | None = None
        self._fs_session: ClientSession | None = None
        self._git_session: ClientSession | None = None

    @property
    def has_filesystem(self) -> bool:
        return self._fs_session is not None

    @property
    def has_git(self) -> bool:
        return self._git_session is not None

    async def connect(self) -> None:
        """Start MCP servers and initialize client sessions."""
        self._exit_stack = AsyncExitStack()

        # Filesystem server
        if self._allowed_dirs:
            fs_params = StdioServerParameters(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem"] + self._allowed_dirs,
            )
            read, write = await self._exit_stack.enter_async_context(
                stdio_client(fs_params)
            )
            self._fs_session = await self._exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await self._fs_session.initialize()
            logger.info("Filesystem MCP session initialized (allowed: %s)", self._allowed_dirs)

        # Git server (Python package via uvx, not npm)
        if self._repo_path:
            git_params = StdioServerParameters(
                command="uvx",
                args=["mcp-server-git", "--repository", self._repo_path],
            )
            read, write = await self._exit_stack.enter_async_context(
                stdio_client(git_params)
            )
            self._git_session = await self._exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await self._git_session.initialize()
            logger.info("Git MCP session initialized (repo: %s)", self._repo_path)

    async def list_tools(self) -> list[dict]:
        """List tools available across all connected MCP servers.

        Returns:
            List of {"name": str, "description": str, "server": str} dicts.
        """
        tools: list[dict] = []

        if self._fs_session:
            result = await self._fs_session.list_tools()
            for t in result.tools:
                tools.append({
                    "name": t.name,
                    "description": t.description or "",
                    "server": "filesystem",
                })

        if self._git_session:
            result = await self._git_session.list_tools()
            for t in result.tools:
                tools.append({
                    "name": t.name,
                    "description": t.description or "",
                    "server": "git",
                })

        return tools

    def server_for_tool(self, tool_name: str) -> str:
        """Return which server handles a given tool ('filesystem' or 'git')."""
        if tool_name in FILESYSTEM_TOOLS:
            return "filesystem"
        elif tool_name in GIT_TOOLS:
            return "git"
        return "unknown"

    def _get_session(self, tool_name: str) -> tuple[ClientSession, str]:
        """Route tool to the correct MCP server session."""
        if tool_name in FILESYSTEM_TOOLS:
            if not self._fs_session:
                raise RuntimeError(
                    f"Filesystem server not connected (needed for {tool_name}). "
                    "Pass allowed_dirs to MCPClientBridge."
                )
            return self._fs_session, "filesystem"
        elif tool_name in GIT_TOOLS:
            if not self._git_session:
                raise RuntimeError(
                    f"Git server not connected (needed for {tool_name}). "
                    "Pass repo_path to MCPClientBridge."
                )
            return self._git_session, "git"
        else:
            raise ValueError(
                f"Unknown tool '{tool_name}' (valid: {', '.join(sorted(VALID_TOOLS))})"
            )

    async def call_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool call, routing to the correct MCP server.

        Handles arg translation (edit_file format, repo_path injection).

        Args:
            tool_name: One of the 14 valid tools.
            args: Arguments dict as output by the model.

        Returns:
            Text result from the MCP server.

        Raises:
            ValueError: If tool_name is not in VALID_TOOLS.
            RuntimeError: If the required server is not connected.
        """
        self._ensure_connected()

        if tool_name not in VALID_TOOLS:
            raise ValueError(
                f"Unknown tool '{tool_name}' (valid: {', '.join(sorted(VALID_TOOLS))})"
            )

        session, server = self._get_session(tool_name)
        translated_args = _translate_args(tool_name, args, self._repo_path)
        mcp_tool_name = _TOOL_NAME_MAP.get(tool_name, tool_name)

        result = await session.call_tool(mcp_tool_name, arguments=translated_args)

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
        """Shut down all MCP server connections."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._fs_session = None
            self._git_session = None
            logger.info("MCP sessions closed")

    async def __aenter__(self) -> "MCPClientBridge":
        await self.connect()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    def _ensure_connected(self) -> None:
        if self._fs_session is None and self._git_session is None:
            raise RuntimeError("Not connected — call connect() or use async with")
