"""
The Serena Model Context Protocol (MCP) Server
"""

import inspect
import json
import os
import platform
import sys
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any, cast

import yaml
from mcp.server.fastmcp import server
from mcp.server.fastmcp.prompts.base import Message, UserMessage
from mcp.server.fastmcp.server import Context, FastMCP, Settings
from sensai.util import logging
from sensai.util.string import dict_string

from multilspy import SyncLanguageServer
from multilspy.multilspy_config import Language, MultilspyConfig
from multilspy.multilspy_logger import MultilspyLogger
from multilspy.multilspy_types import SymbolKind
from serena.gui_log_viewer import GuiLogViewer, GuiLogViewerHandler
from serena.llm.prompt_factory import PromptFactory
from serena.symbol import SymbolLocation, SymbolManager
from serena.util.file_system import scan_directory
from serena.util.shell import execute_shell_command

log = logging.getLogger(__name__)


def configure_logging(*args, **kwargs) -> None:  # type: ignore
    log_format = "%(levelname)-5s %(asctime)-15s %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    level = logging.INFO

    # configure logging to stderr (will be captured by Claude Desktop); stdio is the MCP communication stream and cannot be used!
    logging.basicConfig(level=level, stream=sys.stderr, format=log_format)

    # configure logging to GUI window
    log_handler = GuiLogViewerHandler(GuiLogViewer(title="Serena Logs"), level=level)
    Logger.root.addHandler(log_handler)


# patch the logging configuration function in fastmcp, because it's hard-coded and broken
server.configure_logging = configure_logging


@dataclass
class SerenaMCPRequestContext:
    language_server: SyncLanguageServer
    project_root: str
    project_config: dict[str, Any]
    prompt_factory: PromptFactory

    def get_serena_managed_dir(self) -> str:
        return os.path.join(self.project_root, ".serena")


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

    log.info(f"Starting serena server for project {project_file}; process id={os.getpid()}, parent process id={os.getppid()}")

    # read project configuration
    with open(project_file, encoding="utf-8") as f:
        project_config = yaml.safe_load(f)
    language = Language(project_config["language"])
    project_root = str(Path(project_config["project_root"]).resolve())

    # create and start the language server instance
    config = MultilspyConfig(code_language=language)
    logger = MultilspyLogger()
    language_server = SyncLanguageServer.create(config, logger, project_root)
    with language_server.start_server():
        yield SerenaMCPRequestContext(
            language_server=language_server, project_root=project_root, project_config=project_config, prompt_factory=PromptFactory()
        )


class MemoriesManager:
    def __init__(self, memory_dir: str):
        self._memory_dir = Path(memory_dir)
        self._memory_dir.mkdir(parents=True, exist_ok=True)

    def _get_memory_file_path(self, memory_file_name: str) -> Path:
        return self._memory_dir / memory_file_name

    def load_memory(self, memory_file_name: str) -> str:
        memory_file_path = self._get_memory_file_path(memory_file_name)
        if not memory_file_path.exists():
            return f"Memory file {memory_file_name} not found, consider creating it with the `write_memory` tool if you need it."
        with open(memory_file_path, encoding="utf-8") as f:
            return f.read()

    def save_memory(self, memory_file_name: str, content: str) -> str:
        memory_file_path = self._get_memory_file_path(memory_file_name)
        with open(memory_file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Memory file {memory_file_name} written."

    def list_memories(self) -> list[str]:
        return [f.name for f in self._memory_dir.iterdir() if f.is_file()]

    def delete_memory(self, memory_file_name: str) -> str:
        memory_file_path = self._get_memory_file_path(memory_file_name)
        memory_file_path.unlink()
        return f"Memory file {memory_file_name} deleted."


mcp_settings = Settings(lifespan=server_lifespan)
mcp = FastMCP(**mcp_settings.model_dump())


class Component(ABC):
    def __init__(self, ctx: Context):
        lifespan_context = cast(SerenaMCPRequestContext, ctx.request_context.lifespan_context)
        self.langsrv = lifespan_context.language_server
        self.project_root = lifespan_context.project_root
        self.project_config = lifespan_context.project_config
        self.prompt_factory = lifespan_context.prompt_factory

        memories_dir = os.path.join(lifespan_context.get_serena_managed_dir(), "memories")
        self.memories_manager = MemoriesManager(memories_dir)


_DEFAULT_MAX_ANSWER_LENGTH = int(2e5)


class Tool(Component):
    @staticmethod
    def _log_tool_application(frame: Any) -> None:
        params = {}
        tool_name = None
        for name, value in frame.f_locals.items():
            if name == "ctx":
                continue
            if name.endswith("Tool"):
                tool_name = name
                continue
            params[name] = value
        log.info(f"{tool_name}: {dict_string(params)}")

    def execute(self, max_answer_chars: int = _DEFAULT_MAX_ANSWER_LENGTH) -> str:
        try:
            self._log_tool_application(inspect.currentframe().f_back)  # type: ignore
            result = self._execute()
            if (n_chars := len(result)) > max_answer_chars:
                result = (
                    f"The answer is too long ({n_chars} characters). "
                    + "Please try a more specific tool query or raise the max_answer_chars parameter."
                )
            log.info(f"Result: {result}")
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
    :param max_answer_chars: if the output is longer than this number of characters,
        no content will be returned. Don't adjust unless there is really no other way to get the content
        required for the task.
    :return: a JSON object with the names of directories and files within the given directory
    """

    class ListDirTool(Tool):
        def _execute(self) -> str:
            dirs, files = scan_directory(
                os.path.join(self.project_root, relative_path),
                relative_to=self.project_root,
                recursive=recursive,
                ignored_dirs=self.project_config["ignored_dirs"],
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

    class GetDirOverviewTool(Tool):
        def _execute(self) -> str:
            return json.dumps(self.langsrv.request_dir_overview(relative_path))

    return GetDirOverviewTool(ctx).execute(max_answer_chars=max_answer_chars)


@mcp.tool()
def get_document_overview(ctx: Context, relative_path: str, max_answer_chars: int = _DEFAULT_MAX_ANSWER_LENGTH) -> str:
    """
    Get an overview of the given file.

    :param ctx: the context object, which will be created and provided automatically
    :param relative_path: the relative path to the file to get the overview of
    :param max_answer_chars: if the overview is longer than this number of characters,
        no content will be returned. Don't adjust unless there is really no other way to get the content
        required for the task.
    :return: a JSON object with the list of tuples (name, kind, line, column) of all top-level symbols in the file.
    """

    class GetDocumentOverviewTool(Tool):
        def _execute(self) -> str:
            return json.dumps(self.langsrv.request_document_overview(relative_path))

    return GetDocumentOverviewTool(ctx).execute(max_answer_chars=max_answer_chars)


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
    Retrieves information on all symbols/code entities, i.e. classes, methods, attributes, variables, etc.
    with the given name.
    The returned symbol location information can subsequently be used to edit the returned symbols
    or to retrieve further information using other tools.
    If you already anticipate that you will need to reference children of the symbol, you can specify a
    depth > 0.

    :param ctx: the context object, which will be created and provided automatically
    :param name: the name of the symbols to find
    :param depth: specifies the depth up to which descendants of the symbol are to be retrieved
        (e.g. depth 1 will retrieve methods and attributes for the case where the symbol refers to a class).
        Provide a non-zero depth if you intend to subsequently query symbols that are contained in the
        retrieved symbol.
    :param dir_relative_path: pass a directory relative path to only consider symbols within this directory.
        If None, the entire codebase will be considered.
    :param include_body: whether to include the body of all symbols in the result.
    :param include_kinds: an optional list of ints representing the LSP symbol kind.
        If provided, only symbols of the given kinds will be included in the result.
        Valid kinds:
        1=file, 2=module, 3=namespace, 4=package, 5=class, 6=method, 7=property, 8=field, 9=constructor, 10=enum,
        11=interface, 12=function, 13=variable, 14=constant, 15=string, 16=number, 17=boolean, 18=array, 19=object,
        20=key, 21=null, 22=enum member, 23=struct, 24=event, 25=operator, 26=type parameter
    :param exclude_kinds: If provided, symbols of the given kinds will be excluded from the result.
        Takes precedence over include_kinds.
    :param substring_matching: whether to use substring matching for the symbol name.
        If True, the symbol name will be matched if it contains the given name as a substring.
    :param max_answer_chars: if the output is longer than this number of characters,
        no content will be returned. Don't adjust unless there is really no other way to get the content
        required for the task. Instead, if the output is too long, you should
        make a stricter query.
    :return: a list of symbols (with symbol locations) that match the given name in JSON format
    """

    class FindSymbolTool(Tool):
        def _execute(self) -> str:
            symbols = SymbolManager(self.langsrv).find_by_name(
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
    Finds symbols that reference the symbol at the given location.
    Note that this function can be used to find subclasses of a class, as subclasses are referencing symbols
    that have the kind class.

    :param ctx: the context object, which will be created and provided automatically
    :param relative_path: the relative path to the file containing the symbol
    :param line: the line number
    :param column: the column
    :param include_body: whether to include the body of the symbols in the result
    :param include_kinds: an optional list of integers representing the LSP symbol kinds to include.
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
            symbols = SymbolManager(self.langsrv).find_referencing_symbols(
                SymbolLocation(relative_path, line, column),
                include_body=include_body,
                include_kinds=include_kinds,
                exclude_kinds=exclude_kinds,
            )
            symbol_dicts = [s.to_dict(kind=True, location=True, depth=0, include_body=include_body) for s in symbols]
            return json.dumps(symbol_dicts)

    return FindReferencingSymbolsTool(ctx).execute(max_answer_chars=max_answer_chars)


@mcp.tool()
def replace_symbol_body(
    ctx: Context,
    relative_path: str,
    line: int,
    column: int,
    body: str,
) -> str:
    """
    Replaces the body of the symbol at the given location.
    Important: Do not try to guess symbol locations but use the find_symbol tool to get the correct location.

    :param ctx: the context object, which will be created and provided automatically
    :param relative_path: the relative path to the file containing the symbol
    :param line: the line number
    :param column: the column
    :param body: the new symbol body. Important: Provide the correct level of indentation
        (as the original body). Note that the first line must not be indented (i.e. no leading spaces).
    """

    class ReplaceSymbolBodyTool(Tool):
        def _execute(self) -> str:
            SymbolManager(self.langsrv).replace_body(
                SymbolLocation(relative_path, line, column),
                body=body,
            )
            return "OK"

    return ReplaceSymbolBodyTool(ctx).execute()


@mcp.tool()
def insert_after_symbol(
    ctx: Context,
    relative_path: str,
    line: int,
    column: int,
    body: str,
) -> str:
    """
    Inserts the given body/content after the end of the definition of the given symbol (via the symbol's location).

    :param ctx: the context object, which will be created and provided automatically
    :param relative_path: the relative path to the file containing the symbol
    :param line: the line number
    :param column: the column
    :param body: the body/content to be inserted
    """

    class AppendAfterSymbolTool(Tool):
        def _execute(self) -> str:
            SymbolManager(self.langsrv).insert_after(
                SymbolLocation(relative_path, line, column),
                body=body,
            )
            return "OK"

    return AppendAfterSymbolTool(ctx).execute()


@mcp.tool()
def insert_before_symbol(
    ctx: Context,
    relative_path: str,
    line: int,
    column: int,
    body: str,
) -> str:
    """
    Inserts the given body/content before the beginning of the definition of the given symbol (via the symbol's location).

    :param ctx: the context object, which will be created and provided automatically
    :param relative_path: the relative path to the file containing the symbol
    :param line: the line number
    :param column: the column
    :param body: the body/content to be inserted
    """

    class AppendAfterSymbolTool(Tool):
        def _execute(self) -> str:
            SymbolManager(self.langsrv).insert_before(
                SymbolLocation(relative_path, line, column),
                body=body,
            )
            return "OK"

    return AppendAfterSymbolTool(ctx).execute()


def delete_lines(
    ctx: Context,
    relative_path: str,
    start_line: int,
    end_line: int,
) -> str:
    """
    Inserts the given content at the given line in the file.

    :param ctx: the context object, which will be created and provided automatically
    :param relative_path: the relative path to the file
    :param start_line: the 0-based index of the first line to be deleted
    :param end_line: the 0-based index of the last line to be deleted
    """

    class DeleteLinesTool(Tool):
        def _execute(self) -> str:
            SymbolManager(self.langsrv).delete_lines(relative_path, start_line, end_line)
            return "OK"

    return DeleteLinesTool(ctx).execute()


def insert_at_line(
    ctx: Context,
    relative_path: str,
    line: int,
    content: str,
) -> str:
    """
    Inserts the given content at the given line in the file.

    :param ctx: the context object, which will be created and provided automatically
    :param relative_path: the relative path to the file
    :param line: the 0-based index of the line to insert content at
    :param content: the body/content to be inserted
    """

    class InsertAtLineTool(Tool):
        def _execute(self) -> str:
            SymbolManager(self.langsrv).insert_at_line(relative_path, line, content)
            return "OK"

    return InsertAtLineTool(ctx).execute()


@mcp.tool()
def check_onboarding_performed(ctx: Context) -> str:
    """
    Check if onboarding was performed yet.
    You should always call this tool in the beginning of the conversation,
    before any question about code or the project is asked.
    You will call this tool only once per conversation.
    """
    memories = json.loads(list_memories(ctx))
    if len(memories) == 0:
        return (
            "Onboarding not performed yet (no memories available). "
            + "You should perform onboarding by calling the `onboarding` tool before proceeding with the task."
        )
    else:
        return "Onboarding already performed, no need to perform it again."


@mcp.tool()
def onboarding(ctx: Context) -> str:
    """
    Call this tool if onboarding was not performed yet.
    You will call this tool at most once per conversation.

    :param ctx: the context object, which will be created and provided automatically
    :return: instructions on how to create the onboarding information
    """
    system = platform.system()

    class OnboardingPrompt(SimplePrompt):
        def _create_prompt(self) -> str:
            return self.prompt_factory.create_onboarding_prompt(system=system)

    return OnboardingPrompt(ctx).create()


@mcp.tool()
def write_memory(ctx: Context, memory_file_name: str, content: str, max_answer_chars: int = _DEFAULT_MAX_ANSWER_LENGTH) -> str:
    """
    Write some general information about this project that can be useful for future tasks to a memory file.
    The information should be short and to the point.
    The memory file name should be meaningful, such that from the name you can infer what the information is about.
    It is better to have multiple small memory files than to have a single large one because
    memories will be read one by one and we only ever want to read relevant memories.

    This tool is either called during the onboarding process or when you have identified
    something worth remembering about the project from the past conversation.
    """
    if len(content) > max_answer_chars:
        raise ValueError(
            f"Content for {memory_file_name    } is too long. Max length is {max_answer_chars} characters. "
            + "Please make the content shorter."
        )

    class WriteMemoryTool(Tool):
        def _execute(self) -> str:
            return self.memories_manager.save_memory(memory_file_name, content)

    return WriteMemoryTool(ctx).execute()


@mcp.tool()
def read_memory(ctx: Context, memory_file_name: str, max_answer_chars: int = _DEFAULT_MAX_ANSWER_LENGTH) -> str:
    """
    Read the content of a memory file. This tool should only be used if the information
    is relevant to the current task. You should be able to infer whether the information
    is relevant from the memory file name.
    You should not read the same memory file multiple times in the same conversation.
    """

    class ReadMemoryTool(Tool):
        def _execute(self) -> str:
            return self.memories_manager.load_memory(memory_file_name)

    return ReadMemoryTool(ctx).execute(max_answer_chars=max_answer_chars)


@mcp.tool()
def list_memories(ctx: Context) -> str:
    """
    List available memories. Any memory can be read using the `read_memory` tool.
    """

    class ListMemoriesTool(Tool):
        def _execute(self) -> str:
            return json.dumps(self.memories_manager.list_memories())

    return ListMemoriesTool(ctx).execute()


@mcp.tool()
def delete_memory(ctx: Context, memory_file_name: str) -> str:
    """
    Delete a memory file. Should only happen if a user asks for it explicitly,
    for example by saying that the information retrieved from a memory file is no longer correct
    or no longer relevant for the project.
    """

    class DeleteMemoryTool(Tool):
        def _execute(self) -> str:
            return self.memories_manager.delete_memory(memory_file_name)

    return DeleteMemoryTool(ctx).execute()


@mcp.tool()
def think_about_collected_information(ctx: Context) -> str:
    """
    Think about the collected information and whether it is sufficient and relevant.
    """

    class ThinkAboutCollectedInformationTool(Tool):
        def _execute(self) -> str:
            return self.prompt_factory.create_think_about_collected_information()

    return ThinkAboutCollectedInformationTool(ctx).execute()


@mcp.tool()
def think_about_task_adherence(ctx: Context) -> str:
    """
    Think about the task at hand and whether you are still on track.
    Especially important if the conversation has been going on for a while and there
    has been a lot of back and forth.
    """

    class ThinkAboutTaskAdherenceTool(Tool):
        def _execute(self) -> str:
            return self.prompt_factory.create_think_about_task_adherence()

    return ThinkAboutTaskAdherenceTool(ctx).execute()


@mcp.tool()
def think_about_whether_you_are_done(ctx: Context) -> str:
    """
    Think about whether you are done with the task.
    """

    class ThinkAboutWhetherYouAreDoneTool(Tool):
        def _execute(self) -> str:
            return self.prompt_factory.create_think_about_whether_you_are_done()

    return ThinkAboutWhetherYouAreDoneTool(ctx).execute()


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
    Search for a pattern in all code files in the project.

    :param ctx: the context object, which will be created and provided automatically
    :param pattern: Regular expression pattern to search for, either as a compiled Pattern or string
    :param context_lines_before: Number of lines of context to include before each match
    :param context_lines_after: Number of lines of context to include after each match
    :param paths_include_glob: optional glob pattern specifying files to include in the search; if not provided, search globally.
    :param paths_exclude_glob: optional glob pattern specifying files to exclude from the search (takes precedence over paths_include_glob).
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


@mcp.tool()
def shell_command(
    ctx: Context,
    command: str,
    cwd: str | None = None,
    capture_stderr: bool = True,
    max_answer_chars: int = _DEFAULT_MAX_ANSWER_LENGTH,
) -> str:
    """
    Execute a shell command and return its output.
    You should have at least once looked at the suggested shell commands from the corresponding memory
    created during the onboarding process before using this tool.
    Never execute unsafe shell commands like `rm -rf /` or similar! Generally be very careful with deletions.

    :param ctx: the context object, which will be created and provided automatically
    :param command: the shell command to execute
    :param cwd: the working directory to execute the command in. If None, the project root will be used.
    :param capture_stderr: whether to capture and return stderr output
    :param max_answer_chars: if the output is longer than this number of characters,
        no content will be returned. Don't adjust unless there is really no other way to get the content
        required for the task.
    :return: a JSON object containing the command's stdout and optionally stderr output
    """

    class ExecuteShellCommandTool(Tool):
        def _execute(self) -> str:
            _cwd = cwd or self.project_root
            result = execute_shell_command(command, cwd=_cwd, capture_stderr=capture_stderr)
            return result.json()

    return ExecuteShellCommandTool(ctx).execute(max_answer_chars=max_answer_chars)
