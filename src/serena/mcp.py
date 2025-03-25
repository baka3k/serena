"""
The Serena Model Context Protocol (MCP) Server
"""

import json
import os
import sys
import traceback
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, cast

import yaml
from mcp.server.fastmcp import server
from mcp.server.fastmcp.prompts.base import Message, UserMessage
from mcp.server.fastmcp.server import Context, FastMCP, Settings
from sensai.util import logging

from multilspy import SyncLanguageServer
from multilspy.multilspy_config import Language, MultilspyConfig
from multilspy.multilspy_logger import MultilspyLogger
from serena.llm.prompt_factory import PromptFactory
from serena.symbol import SymbolRetriever
from serena.util.file_system import scan_directory

log = logging.getLogger(__name__)


def configure_logging(*args, **kwargs) -> None:  # type: ignore
    # log to stderr (will be captured by Claude Desktop); stdio is the MCP communication stream and cannot be used!
    logging.basicConfig(
        level=logging.DEBUG, stream=sys.stderr, format="%(levelname)-5s %(asctime)-15s %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    )


# patch the logging configuration function in fastmcp, because it's hard-coded and broken
server.configure_logging = configure_logging


@dataclass
class SerenaMCPRequestContext:
    language_server: SyncLanguageServer
    project_root: str
    project_config: dict[str, Any]
    prompt_factory: PromptFactory


@asynccontextmanager
async def server_lifespan(mcp_server: FastMCP) -> AsyncIterator[SerenaMCPRequestContext]:
    """Manage server startup and shutdown lifecycle."""
    argv = sys.argv[1:]
    if len(argv) != 1:
        print("\nUsage: mcp_server <.yml project file>", file=sys.stderr)
        sys.exit(1)

    project_file = argv[0]
    if not os.path.exists(project_file):
        print(f"Project file not found: {project_file}", file=sys.stderr)
        sys.exit(1)

    # read project configuration
    with open(project_file, encoding="utf-8") as f:
        project_config = yaml.safe_load(f)
    language = Language(project_config["language"])
    project_root = project_config["project_root"]

    # create and start the language server instance
    config = MultilspyConfig(code_language=language)
    logger = MultilspyLogger()
    language_server = SyncLanguageServer.create(config, logger, project_root)
    language_server.start()
    try:
        yield SerenaMCPRequestContext(
            language_server=language_server, project_root=project_root, project_config=project_config, prompt_factory=PromptFactory()
        )
    finally:
        language_server.stop()


mcp_settings = Settings(lifespan=server_lifespan)
mcp = FastMCP(**mcp_settings.model_dump())


class Component(ABC):
    def __init__(self, ctx: Context):
        lifespan_context = cast(SerenaMCPRequestContext, ctx.request_context.lifespan_context)
        self.langsrv = lifespan_context.language_server
        self.project_root = lifespan_context.project_root
        self.project_config = lifespan_context.project_config
        self.prompt_factory = lifespan_context.prompt_factory


class Tool(Component):
    def execute(self) -> str:
        try:
            return self._execute()
        except Exception as e:
            msg = f"Error executing tool: {e}\n{traceback.format_exc()}"
            return msg

    @abstractmethod
    def _execute(self) -> str:
        pass


class SimplePrompt(Component):
    def create(self) -> str:
        return self._create_prompt()

    @abstractmethod
    def _create_prompt(self) -> str:
        pass


class SequentialPrompt(Component):
    def __init__(self, ctx: Context):
        super().__init__(ctx)
        self.messages: list[Message] = []

    def create(self) -> list[Message]:
        self._add_messages()
        return self.messages

    @abstractmethod
    def _add_messages(self) -> None:
        pass

    def _add_user_message(self, msg: str) -> None:
        self.messages.append(UserMessage(content=msg))


@mcp.tool()
def read_file(ctx: Context, relative_path: str) -> str:
    """
    :param ctx: the context object, which will be created and provided automatically
    :param relative_path: the relative path to the file to read
    :return: the full text of the file at the given relative path
    """
    log.info(f"read_file: {relative_path=}")

    class ReadFileTool(Tool):
        def _execute(self) -> str:
            return self.langsrv.retrieve_full_file_content(relative_path)

    return ReadFileTool(ctx).execute()


@mcp.tool()
def create_text_file(ctx: Context, relative_path: str, content: str) -> str:
    """
    :param ctx: the context object, which will be created and provided automatically
    :param relative_path: the relative path to the file to create
    :param content: the (utf-8-encoded) content to write to the file
    :return: a message indicating success or failure
    """
    log.info(f"create_file: {relative_path=}")

    class CreateFileTool(Tool):
        def _execute(self) -> str:
            absolute_path = os.path.join(self.project_root, relative_path)
            with open(absolute_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"File created: {relative_path}"

    return CreateFileTool(ctx).execute()


@mcp.tool()
def list_dir(ctx: Context, relative_path: str, recursive: bool) -> str:
    """
    :param ctx: the context object, which will be created and provided automatically
    :param relative_path: the relative path to the directory to list; pass "." to scan the project root
    :param recursive: whether to scan subdirectories recursively
    :return: a JSON object with the names of directories and files within the given directory
    """
    log.info(f"list_dir: {relative_path=}")

    class ListDirTool(Tool):
        def _execute(self) -> str:
            dirs, files = scan_directory(
                os.path.join(self.project_root, relative_path), recursive=recursive, ignored_dirs=self.project_config["ignored_dirs"]
            )
            return json.dumps({"dirs": dirs, "files": files})

    return ListDirTool(ctx).execute()


@mcp.tool()
def get_dir_overview(ctx: Context, relative_path: str) -> str:
    """
    Get an overview of the given directory.
    For each file in the directory, we list the top-level symbols in the file (name, kind, line).

    :param ctx: the context object, which will be created and provided automatically
    :param relative_path: the relative path to the directory to get the overview of
    :return: a JSON object mapping relative paths of all contained files to info about top-level symbols in the file (name, kind, line).
    """
    log.info(f"get_dir_overview: {relative_path=}")

    class GetDirOverviewTool(Tool):
        def _execute(self) -> str:
            return json.dumps(self.langsrv.request_dir_overview(relative_path))

    return GetDirOverviewTool(ctx).execute()


@mcp.tool()
def find_symbol(ctx: Context, name: str, depth: int = 0) -> str:
    """
    Retrieves information on the symbol/code entity with the given name

    :param ctx: the context object, which will be created and provided automatically
    :param name: the name of the symbol
    :param depth: specifies the depth up to which descendants of the symbol are to be retrieved
        (e.g. depth 1 will retrieve methods for the case where the symbol refers to a class)
    :return: a list of JSON objects with the result
    """

    class FindSymbolTool(Tool):
        def _execute(self) -> str:
            symbols = SymbolRetriever(self.langsrv).find(name)
            symbol_dicts = [s.to_dict(kind=True, location=True, depth=depth) for s in symbols]
            return json.dumps(symbol_dicts)

    return FindSymbolTool(ctx).execute()


@mcp.tool()
def find_referencing_symbols(ctx: Context, relative_path: str, line: int, column: int) -> str:
    """
    :param ctx: the context object, which will be created and provided automatically
    :param relative_path: the relative path to the file containing the symbol
    :param line: the line number
    :param column: the column
    :return: a list of JSON objects with the symbols referencing the requested symbol
    """

    class FindReferencingSymbolsTool(Tool):
        def _execute(self) -> str:
            symbols = SymbolRetriever(self.langsrv).find_references(relative_path, line, column)
            symbol_dicts = [s.to_dict(kind=True, location=True, depth=0) for s in symbols]
            return json.dumps(symbol_dicts)

    return FindReferencingSymbolsTool(ctx).execute()


@mcp.tool()
def onboarding(ctx: Context) -> str:
    """
    :param ctx: the context object, which will be created and provided automatically
    :return: instructions on how to create the onboarding information
    """
    onboarding_file = "serena_onboarding.md"

    class OnboardingPrompt(SimplePrompt):
        def _create_prompt(self) -> str:
            return self.prompt_factory.create_onboarding_prompt(onboarding_file=onboarding_file)

    return OnboardingPrompt(ctx).create()
