import os
from pathlib import Path
from typing import Literal

import click
from sensai.util import logging
from tqdm import tqdm

from serena.agent import SerenaAgent
from serena.config.serena_config import SerenaConfig
from serena.constants import DEFAULT_CONTEXT, DEFAULT_MODES
from serena.mcp import SerenaMCPFactorySingleProcess
from serena.project import Project

log = logging.getLogger(__name__)


class ProjectType(click.ParamType):
    name = "[PROJECT_NAME|PROJECT_PATH]"

    def convert(self, value: str, param: click.Parameter | None, ctx: click.Context | None) -> str:
        path = Path(value).resolve()
        if path.exists() and path.is_dir():
            return str(path)  # Valid path
        return value  # Assume it's a project name


PROJECT_TYPE = ProjectType()


@click.command()
@click.option(
    "--project",
    "project",
    type=PROJECT_TYPE,
    default=None,
    help="Either an absolute path to the project directory or a name of an already registered project. "
    "If the project passed here hasn't been registered yet, it will be registered automatically and can be activated by its name afterwards.",
)
# Keep --project-file for backwards compatibility
@click.option(
    "--project-file",
    "project",  # Use same destination variable to avoid conflicts
    type=PROJECT_TYPE,
    default=None,
    help="[DEPRECATED] Use --project instead.",
)
# Positional argument for backwards compatibility
@click.argument(
    "project_file_arg",
    type=PROJECT_TYPE,
    required=False,
    default=None,
    metavar="",  # don't display anything since it's deprecated
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
    help="Transport protocol. If you start the server yourself (as opposed to an MCP Client starting the server), sse is recommended.",
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
@click.option(
    "--enable-web-dashboard",
    type=bool,
    is_flag=False,
    default=None,
    help="Whether to enable the web dashboard. If not specified, will take the value from the serena configuration.",
)
@click.option(
    "--enable-gui-log-window",
    type=bool,
    is_flag=False,
    default=None,
    help="Whether to enable the GUI log window. It currently does not work on macOS, and setting this to True will be ignored then. "
    "If not specified, will take the value from the serena configuration.",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    default=None,
    help="Log level for GUI, dashboard and other logging. If not specified, will take the value from the serena configuration.",
)
@click.option(
    "--trace-lsp-communication",
    type=bool,
    is_flag=False,
    default=None,
    help="Whether to trace the communication between Serena and the language servers. This is useful for debugging language server issues.",
)
@click.option(
    "--tool-timeout",
    type=float,
    default=None,
    help="Timeout in seconds for tool execution. If not specified, will take the value from the serena configuration.",
)
def start_mcp_server(
    project: str | None,
    project_file_arg: str | None,
    context: str = DEFAULT_CONTEXT,
    modes: tuple[str, ...] = DEFAULT_MODES,
    transport: Literal["stdio", "sse"] = "stdio",
    host: str = "0.0.0.0",
    port: int = 8000,
    enable_web_dashboard: bool | None = None,
    enable_gui_log_window: bool | None = None,
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None = None,
    trace_lsp_communication: bool | None = None,
    tool_timeout: float | None = None,
) -> None:
    """Starts the Serena MCP server. By default, will not activate any project at startup.
    If you want to start with an already active project, use --project to pass the project name or path.

    Use --context to specify the execution environment and --mode to specify behavior mode(s).
    The modes may be adjusted after startup (via the corresponding tool), but the context cannot be changed.
    """
    # Prioritize the positional argument if provided
    # This is for backward compatibility with the old CLI, should be removed in the future!
    project_file = project_file_arg if project_file_arg is not None else project

    mcp_factory = SerenaMCPFactorySingleProcess(context=context, project=project_file)
    mcp_server = mcp_factory.create_mcp_server(
        host=host,
        port=port,
        modes=modes,
        enable_web_dashboard=enable_web_dashboard,
        enable_gui_log_window=enable_gui_log_window,
        log_level=log_level,
        trace_lsp_communication=trace_lsp_communication,
        tool_timeout=tool_timeout,
    )

    # log after server creation such that the log appears in the GUI
    if project_file_arg is not None:
        log.warning(
            "The positional argument for the project file path is deprecated and will be removed in the future! "
            "Please pass the project file path via the `--project` option instead.\n"
            f"Used path: {project_file}"
        )

    log.info("Starting MCP server ...")

    mcp_server.run(transport=transport)


@click.command()
@click.argument("project", type=click.Path(exists=True), required=False, default=os.getcwd())
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]), default="WARNING")
@click.option(
    "--only-instructions",
    is_flag=True,
    help="If set, only print the initial instructions prompt (without prefix or postfix). The prefix and postfix are small additions that are used to"
    "make the initial instructions more digestible for an LLM in the context of a system prompt, as opposed to their normal usage within a conversation.",
)
def print_system_prompt(project: str, log_level: str = "WARNING", only_instructions: bool = False) -> None:
    """
    Print the system prompt (initial instructions with a potential pre/post-fix) for a project.
    """
    prefix_to_instructions_prompt = (
        "You will receive access to Serena's symbolic tools. Below are instructions for using them, take them into account."
    )
    postfix_to_instructions_prompt = "You begin the conversation by acknowledging that you understood the above instructions on the symbolic tools and are ready to receive a task. You will not need to call the tool on getting initial instructions."

    from serena.tools.workflow_tools import InitialInstructionsTool

    log_level_int = logging.getLevelNamesMapping()[log_level.upper()]
    logging.configure(level=log_level_int)

    agent = SerenaAgent(project=os.path.abspath(project), serena_config=SerenaConfig(web_dashboard=False, log_level=log_level_int))
    initial_instructions_tool = agent.get_tool(InitialInstructionsTool)
    initial_instructions = initial_instructions_tool.apply()
    if only_instructions:
        print(initial_instructions)
    else:
        system_prompt = f"{prefix_to_instructions_prompt}\n{initial_instructions}\n{postfix_to_instructions_prompt}"
        print(system_prompt)


@click.command()
@click.argument("project", type=click.Path(exists=True), required=False, default=os.getcwd())
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]), default="WARNING")
def index_project(project: str, log_level: str = "INFO") -> None:
    """
    Index a project by saving the symbols of files to Serena's language server cache.

    :param project: the project to index. By default, the current working directory is used.
    """
    log_level_int = logging.getLevelNamesMapping()[log_level.upper()]
    project_instance = Project.load(os.path.abspath(project))
    print(f"Indexing symbols in project {project}")
    ls = project_instance.create_language_server(log_level=log_level_int)
    save_after_n_files = 10
    with ls.start_server():
        parsed_files = project_instance.gather_source_files()
        files_processed = 0
        pbar = tqdm(parsed_files, disable=False)
        for relative_file_path in pbar:
            pbar.set_description(f"Indexing ({os.path.basename(relative_file_path)})")
            ls.request_document_symbols(relative_file_path, include_body=False)
            ls.request_document_symbols(relative_file_path, include_body=True)
            files_processed += 1
            if files_processed % save_after_n_files == 0:
                ls.save_cache()
        ls.save_cache()
    print(f"Symbols saved to {ls.cache_path}")
