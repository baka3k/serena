"""
The Serena Model Context Protocol (MCP) Server
"""

import json
import os
import sys
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import AsyncIterator, Sequence
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
from multilspy.multilspy_types import SymbolKind
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

    print(f"Starting serena server for project {project_file}")

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


_DEFAULT_MAX_ANSWER_LENGTH = int(2e5)


class Tool(Component):
    _on_too_long_answer_msg = """
The answer is too long to display ({} characters).
Please try a more specific tool query or raise the max_answer_length parameter.
"""

    def execute(self, max_answer_chars: int = _DEFAULT_MAX_ANSWER_LENGTH) -> str:
        try:
            result = self._execute()
            if (n_chars := len(result)) > max_answer_chars:
                return self._on_too_long_answer_msg.format(n_chars)
            return result
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
def read_file(
    ctx: Context, relative_path: str, start_line: int = 0, end_line: int | None = None, max_answer_chars: int = _DEFAULT_MAX_ANSWER_LENGTH
) -> str:
    """
    Read the given file or a chunk of it (between start_line and end_line).

    :param ctx: the context object, which will be created and provided automatically
    :param relative_path: the relative path to the file to read
    :param start_line: the start line of the range to read
    :param end_line: the end line of the range to read. If None, the entire file will be read.
    :param max_answer_chars: if the file (chunk) is longer than this number of characters,
        no content will be returned. Don't adjust unless there is really no other way to get the content
        required for the task.
    :return: the full text of the file at the given relative path
    """
    log.info(f"read_file: {relative_path=}")

    class ReadFileTool(Tool):
        def _execute(self) -> str:
            result = self.langsrv.retrieve_full_file_content(relative_path)
            result_lines = result.splitlines()
            if end_line is None:
                result_lines = result_lines[start_line:]
            else:
                result_lines = result_lines[start_line:end_line]
            result = "\n".join(result_lines)
            return result

    return ReadFileTool(ctx).execute(max_answer_chars=max_answer_chars)


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
def list_dir(ctx: Context, relative_path: str, recursive: bool, max_answer_chars: int = _DEFAULT_MAX_ANSWER_LENGTH) -> str:
    """
    :param ctx: the context object, which will be created and provided automatically
    :param relative_path: the relative path to the directory to list; pass "." to scan the project root
    :param recursive: whether to scan subdirectories recursively
    :param max_answer_chars: if the directory is longer than this number of characters,
        no content will be returned. Don't adjust unless there is really no other way to get the content
        required for the task.
    :return: a JSON object with the names of directories and files within the given directory
    """
    log.info(f"list_dir: {relative_path=}")

    class ListDirTool(Tool):
        def _execute(self) -> str:
            dirs, files = scan_directory(
                os.path.join(self.project_root, relative_path), recursive=recursive, ignored_dirs=self.project_config["ignored_dirs"]
            )
            return json.dumps({"dirs": dirs, "files": files})

    return ListDirTool(ctx).execute(max_answer_chars=max_answer_chars)


@mcp.tool()
def get_dir_overview(ctx: Context, relative_path: str, max_answer_chars: int = _DEFAULT_MAX_ANSWER_LENGTH) -> str:
    """
    Get an overview of the given directory.
    For each file in the directory, we list the top-level symbols in the file (name, kind, line).

    :param ctx: the context object, which will be created and provided automatically
    :param relative_path: the relative path to the directory to get the overview of
    :param max_answer_chars: if the overview is longer than this number of characters,
        no content will be returned. Don't adjust unless there is really no other way to get the content
        required for the task. If the overview is too long, you should use a smaller directory instead,
        (e.g. a subdirectory).
    :return: a JSON object mapping relative paths of all contained files to info about top-level symbols in the file (name, kind, line).
    """
    log.info(f"get_dir_overview: {relative_path=}")

    class GetDirOverviewTool(Tool):
        def _execute(self) -> str:
            return json.dumps(self.langsrv.request_dir_overview(relative_path))

    return GetDirOverviewTool(ctx).execute(max_answer_chars=max_answer_chars)


@mcp.tool()
def find_symbol(
    ctx: Context,
    name: str,
    depth: int = 0,
    include_body: bool = False,
    include_kinds: Sequence[SymbolKind] | None = None,
    exclude_kinds: Sequence[SymbolKind] | None = None,
    substring_matching: bool = False,
    dir_relative_path: str | None = None,
    max_answer_chars: int = _DEFAULT_MAX_ANSWER_LENGTH,
) -> str:
    """
    Retrieves information on all symbols/code entities with the given name.

    :param ctx: the context object, which will be created and provided automatically
    :param name: the name of the symbols to find
    :param dir_relative_path: pass a directory relative path to only consider symbols within this directory.
        If None, the entire codebase will be considered.
    :param include_body: whether to include the body of all symbols in the result.
        Note: you can filter out the bodies of the children if you set include_children_body=False
        in the to_dict method.
    :param include_kinds: an optional sequence of ints representing the LSP symbol kind.
        If provided, only symbols of the given kinds will be included in the result.
    :param exclude_kinds: If provided, symbols of the given kinds will be excluded from the result.
        Takes precedence over include_kinds.
    :param substring_matching: whether to use substring matching for the symbol name.
        If True, the symbol name will be matched if it contains the given name as a substring.
    :param max_answer_chars: if the output is longer than this number of characters,
        no content will be returned. Don't adjust unless there is really no other way to get the content
        required for the task. Instead, if the output is too long, you should
        make a stricter query.
    :return: a list of symbols that match the given name
    """

    class FindSymbolTool(Tool):
        def _execute(self) -> str:
            symbols = SymbolRetriever(self.langsrv).find(
                name,
                include_body=include_body,
                include_kinds=include_kinds,
                exclude_kinds=exclude_kinds,
                substring_matching=substring_matching,
                dir_relative_path=dir_relative_path,
            )
            symbol_dicts = [s.to_dict(kind=True, location=True, depth=depth, include_body=include_body) for s in symbols]
            return json.dumps(symbol_dicts)

    return FindSymbolTool(ctx).execute(max_answer_chars=max_answer_chars)


@mcp.tool()
def find_referencing_symbols(
    ctx: Context,
    relative_path: str,
    line: int,
    column: int,
    include_body: bool = False,
    include_kinds: Sequence[SymbolKind] | None = None,
    exclude_kinds: Sequence[SymbolKind] | None = None,
    max_answer_chars: int = _DEFAULT_MAX_ANSWER_LENGTH,
) -> str:
    """
    :param ctx: the context object, which will be created and provided automatically
    :param relative_path: the relative path to the file containing the symbol
    :param line: the line number
    :param column: the column
    :param include_body: whether to include the body of the symbols in the result
    :param include_kinds: an optional sequence of ints representing the LSP symbol kind.
        If provided, only symbols of the given kinds will be included in the result.
    :param exclude_kinds: If provided, symbols of the given kinds will be excluded from the result.
        Takes precedence over include_kinds.
    :param max_answer_chars: if the output is longer than this number of characters,
        no content will be returned. Don't adjust unless there is really no other way to get the content
        required for the task. Instead, if the output is too long, you should
        make a stricter query.
    :return: a list of JSON objects with the symbols referencing the requested symbol
    """

    class FindReferencingSymbolsTool(Tool):
        def _execute(self) -> str:
            symbols = SymbolRetriever(self.langsrv).find_references(
                relative_path, line, column, include_body=include_body, include_kinds=include_kinds, exclude_kinds=exclude_kinds
            )
            symbol_dicts = [s.to_dict(kind=True, location=True, depth=0, include_body=include_body) for s in symbols]
            return json.dumps(symbol_dicts)

    return FindReferencingSymbolsTool(ctx).execute(max_answer_chars=max_answer_chars)


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


@mcp.tool()
def search_files_for_pattern(
    ctx: Context,
    pattern: str,
    context_lines_before: int = 0,
    context_lines_after: int = 0,
    paths_include_glob: str | None = None,
    paths_exclude_glob: str | None = None,
    max_answer_chars: int = _DEFAULT_MAX_ANSWER_LENGTH,
) -> str:
    """
    Search for a pattern in all codefiles in the project.

    :param ctx: the context object, which will be created and provided automatically
    :param pattern: Regular expression pattern to search for, either as a compiled Pattern or string
    :param context_lines_before: Number of lines of context to include before each match
    :param context_lines_after: Number of lines of context to include after each match
    :param paths_include_glob: Glob pattern to filter which files to include in the search
    :param paths_exclude_glob: Glob pattern to filter which files to exclude from the search. Takes precedence over paths_include_glob.
    :param max_answer_chars: if the output is longer than this number of characters,
        no content will be returned. Don't adjust unless there is really no other way to get the content
        required for the task. Instead, if the output is too long, you should
        make a stricter query.
    :return: A JSON object mapping file paths to lists of matched consecutive lines (with context, if requested).

    """

    class SearchInAllCodeTool(Tool):
        def _execute(self) -> str:
            matches = self.langsrv.search_files_for_pattern(
                pattern=pattern,
                context_lines_before=context_lines_before,
                context_lines_after=context_lines_after,
                paths_include_glob=paths_include_glob,
                paths_exclude_glob=paths_exclude_glob,
            )
            # group matches by file
            file_to_matches: dict[str, list[str]] = defaultdict(list)
            for match in matches:
                assert match.source_file_path is not None
                file_to_matches[match.source_file_path].append(match.to_display_string())
            return json.dumps(file_to_matches)

    return SearchInAllCodeTool(ctx).execute(max_answer_chars=max_answer_chars)
