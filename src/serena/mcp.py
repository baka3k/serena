"""
The Serena Model Context Protocol (MCP) Server
"""

import sys
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from logging import Formatter, Logger, StreamHandler
from typing import Any, Literal

import click  # Add click import
import docstring_parser
from mcp.server.fastmcp import server
from mcp.server.fastmcp.server import FastMCP, Settings
from mcp.server.fastmcp.tools.base import Tool as MCPTool
from mcp.server.fastmcp.utilities.func_metadata import func_metadata
from sensai.util import logging
from sensai.util.helper import mark_used

from serena.agent import SerenaAgent, Tool, show_fatal_exception_safe
from serena.config import SerenaAgentContext, SerenaAgentMode
from serena.constants import DEFAULT_CONTEXT, DEFAULT_MODES

log = logging.getLogger(__name__)
LOG_FORMAT = "%(levelname)-5s %(asctime)-15s %(name)s:%(funcName)s:%(lineno)d - %(message)s"
LOG_LEVEL = logging.INFO


def configure_logging(*args, **kwargs) -> None:  # type: ignore
    # configure logging to stderr (will be captured by Claude Desktop); stdio is the MCP communication stream and cannot be used!
    Logger.root.setLevel(LOG_LEVEL)
    handler = StreamHandler(stream=sys.stderr)
    handler.formatter = Formatter(LOG_FORMAT)
    Logger.root.addHandler(handler)


# patch the logging configuration function in fastmcp, because it's hard-coded and broken
server.configure_logging = configure_logging


@dataclass
class SerenaMCPRequestContext:
    agent: SerenaAgent


def make_tool(
    tool: Tool,
) -> MCPTool:
    func_name = tool.get_name()

    apply_fn = getattr(tool, "apply")
    if apply_fn is None:
        raise ValueError(f"Tool does not have an apply method: {tool}")

    func_doc = apply_fn.__doc__ or ""
    is_async = False

    func_arg_metadata = func_metadata(apply_fn)
    parameters = func_arg_metadata.arg_model.model_json_schema()

    docstring = docstring_parser.parse(func_doc)

    # Mount the tool description as a combination of the docstring description and
    # the return value description, if it exists.
    if docstring.description:
        func_doc = f"{docstring.description.strip().strip('.')}."
    else:
        func_doc = ""
    if (docstring.returns) and (docstring_returns := docstring.returns.description):
        # Only add a space before "Returns" if func_doc is not empty
        prefix = " " if func_doc else ""
        func_doc = f"{func_doc}{prefix}Returns {docstring_returns.strip().strip('.')}."

    # Parse the parameter descriptions from the docstring and add pass its description
    # to the parameters schema.
    docstring_params = {param.arg_name: param for param in docstring.params}
    parameters_properties: dict[str, dict[str, Any]] = parameters["properties"]
    for parameter, properties in parameters_properties.items():
        if (param_doc := docstring_params.get(parameter)) and (param_doc.description):
            param_desc = f"{param_doc.description.strip().strip('.') + '.'}"
            properties["description"] = param_desc[0].upper() + param_desc[1:]

    def execute_fn(**kwargs) -> str:  # type: ignore
        return tool.apply_ex(log_call=True, catch_exceptions=True, **kwargs)

    return MCPTool(
        fn=execute_fn,
        name=func_name,
        description=func_doc,
        parameters=parameters,
        fn_metadata=func_arg_metadata,
        is_async=is_async,
        context_kwarg=None,
    )


def create_mcp_server_and_agent(
    project: str | None,
    host: str = "0.0.0.0",
    port: int = 8000,
    context: str = DEFAULT_CONTEXT,
    modes: Sequence[str] = DEFAULT_MODES,
) -> tuple[FastMCP, SerenaAgent]:
    """
    Create an MCP server.

    :param project: The path to the project directory or the `project.yml` file therein, or None.
    :param host: The host to bind to
    :param port: The port to bind to
    :param context: The context name or path to context file
    :param modes: List of mode names or paths to mode files
    """
    mcp: FastMCP | None = None
    context_instance = SerenaAgentContext.load(context)
    modes_instances = [SerenaAgentMode.load(mode) for mode in modes]

    try:
        agent = SerenaAgent(
            project_config=project,
            # Callback disabled for the time being (see above)
            # project_activation_callback=update_tools
            context=context_instance,
            modes=modes_instances,
        )
    except Exception as e:
        show_fatal_exception_safe(e)
        raise

    def update_tools() -> None:
        """Update the tools in the MCP server."""
        # Tools may change as a result of project activation.
        # NOTE: While we could pass updated tool information on to the MCP server via the callback, Claude Desktop does not,
        # unfortunately, query for changed tools. It only queries for changed resources and prompts regularly,
        # so we need to register all tools at startup, unfortunately.
        nonlocal mcp, agent
        tools = agent.get_exposed_tools()
        if mcp is not None:
            mcp._tool_manager._tools = {}
            for tool in tools:
                # noinspection PyProtectedMember
                mcp._tool_manager._tools[tool.get_name()] = make_tool(tool)

    @asynccontextmanager
    async def server_lifespan(mcp_server: FastMCP) -> AsyncIterator[None]:
        """Manage server startup and shutdown lifecycle."""
        nonlocal agent
        mark_used(mcp_server)
        yield
        del agent

    mcp_settings = Settings(lifespan=server_lifespan, host=host, port=port)
    mcp = FastMCP(**mcp_settings.model_dump())

    update_tools()

    return mcp, agent


@click.command()
# Add --project option as the primary, more intuitive interface
@click.option(
    "--project",
    "project_file_opt",  # Use same destination variable to avoid conflicts
    type=click.Path(exists=True, dir_okay=True, resolve_path=True),
    default=None,
    help="Path to the .yml project file. "
    "Does not need to be provided at startup since you can activate a project later by simply asking the agent to do so "
    "(there is a dedicated tool for this purpose).",
)
# Keep --project-file for backwards compatibility
@click.option(
    "--project-file",
    "project_file_opt",  # Use same destination variable to avoid conflicts
    type=click.Path(exists=True, dir_okay=True, resolve_path=True),
    default=None,
    help="[DEPRECATED] Use --project instead. Optional path to the .yml project file via option."
    "Does not need to be provided at startup since you can activate a project later by simply asking the agent to do so "
    "(there is a dedicated tool for this purpose).",
)
# Positional argument for backwards compatibility
@click.argument(
    "project_file_arg",
    type=click.Path(exists=True, dir_okay=True, resolve_path=True),
    required=False,
    default=None,
)
@click.option(
    "--context",
    type=str,
    show_default=True,
    default=DEFAULT_CONTEXT,
    help="Context to use. This can be a name of a built-in context ('desktop-app', 'agent', 'ide-assistant') "
    "or a path to a custom context YAML file.",
)
@click.option(
    "--mode",
    "modes",
    type=str,
    multiple=True,
    default=DEFAULT_MODES,
    show_default=True,
    help="Mode(s) to use. This can be names of built-in modes ('planning', 'editing', 'one-shot', 'interactive') "
    "or paths to custom mode YAML files. Can be specified multiple times to combine modes.",
)
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    show_default=True,
    help="Transport protocol.",
)
@click.option(
    "--host",
    type=str,
    show_default=True,
    default="0.0.0.0",
    help="Host to bind to (for SSE transport).",
)
@click.option(
    "--port",
    type=int,
    show_default=True,
    default=8000,
    help="Port to bind to (for SSE transport).",
)
def start_mcp_server(
    project_file_opt: str | None,
    project_file_arg: str | None,
    context: str = DEFAULT_CONTEXT,
    modes: tuple[str, ...] = DEFAULT_MODES,
    transport: Literal["stdio", "sse"] = "stdio",
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """Starts the Serena MCP server.

    Accepts a path to the project directory or the `project.yml` file therein via the --project option.

    Use --context to specify the execution environment and --mode to specify behavior mode(s).
    """
    # Prioritize the positional argument if provided
    # This is for backward compatibility with the old CLI, should be removed in the future!
    project_file = project_file_arg if project_file_arg is not None else project_file_opt

    mcp_server, agent = create_mcp_server_and_agent(project=project_file, host=host, port=port, context=context, modes=modes)

    # log after server creation such that the log appears in the GUI
    if project_file_arg is not None:
        log.warning(
            "The positional argument for the project file path is deprecated and will be removed in the future! "
            "Please pass the project file path via the `--project` option instead.\n"
            f"Used path: {project_file}"
        )

    log.info(f"Starting serena agent in MCP server with config:\n{agent.get_current_config_overview()}")

    mcp_server.run(transport=transport)
