import inspect
import json
import os
import platform
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterable
from copy import copy
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self, TypeVar

from mcp.server.fastmcp.utilities.func_metadata import FuncMetadata, func_metadata
from sensai.util import logging
from sensai.util.string import dict_string

from serena.prompt_factory import PromptFactory
from serena.symbol import SymbolManager
from serena.util.inspection import iter_subclasses
from solidlsp import SolidLanguageServer

if TYPE_CHECKING:
    from serena.agent import LinesRead, MemoriesManager, SerenaAgent

log = logging.getLogger(__name__)
T = TypeVar("T")
SUCCESS_RESULT = "OK"


class Component(ABC):
    def __init__(self, agent: "SerenaAgent"):
        self.agent = agent

    @property
    def language_server(self) -> SolidLanguageServer:
        assert self.agent.language_server is not None
        return self.agent.language_server

    def get_project_root(self) -> str:
        """
        :return: the root directory of the active project, raises a ValueError if no active project configuration is set
        """
        return self.agent.get_project_root()

    @property
    def prompt_factory(self) -> PromptFactory:
        return self.agent.prompt_factory

    @property
    def memories_manager(self) -> "MemoriesManager":
        assert self.agent.memories_manager is not None
        return self.agent.memories_manager

    @property
    def symbol_manager(self) -> SymbolManager:
        assert self.agent.symbol_manager is not None
        return self.agent.symbol_manager

    @property
    def lines_read(self) -> "LinesRead":
        assert self.agent.lines_read is not None
        return self.agent.lines_read


TOOL_DEFAULT_MAX_ANSWER_LENGTH = int(2e5)


class ToolMarkerCanEdit:
    """
    Marker class for all tools that can perform editing operations on files.
    """


class ToolMarkerDoesNotRequireActiveProject:
    pass


class ToolInterface(ABC):
    """Protocol defining the complete interface that make_tool() expects from a tool."""

    @abstractmethod
    def get_name(self) -> str:
        """Get the tool name."""
        ...

    @abstractmethod
    def get_apply_docstring(self) -> str:
        """Get the docstring for the tool application, used by the MCP server."""
        ...

    @abstractmethod
    def get_apply_fn_metadata(self) -> FuncMetadata:
        """Get the metadata for the tool application function, used by the MCP server."""
        ...

    @abstractmethod
    def apply_ex(self, log_call: bool = True, catch_exceptions: bool = True, **kwargs: Any) -> str:
        """Apply the tool with logging and exception handling."""
        ...


class Tool(Component, ToolInterface):
    # NOTE: each tool should implement the apply method, which is then used in
    # the central method of the Tool class `apply_ex`.
    # Failure to do so will result in a RuntimeError at tool execution time.
    # The apply method is not declared as part of the base Tool interface since we cannot
    # know the signature of the (input parameters of the) method in advance.
    #
    # The docstring and types of the apply method are used to generate the tool description
    # (which is use by the LLM, so a good description is important)
    # and to validate the tool call arguments.

    @classmethod
    def get_name_from_cls(cls) -> str:
        name = cls.__name__
        if name.endswith("Tool"):
            name = name[:-4]
        # convert to snake_case
        name = "".join(["_" + c.lower() if c.isupper() else c for c in name]).lstrip("_")
        return name

    def get_name(self) -> str:
        return self.get_name_from_cls()

    def get_apply_fn(self) -> Callable:
        apply_fn = getattr(self, "apply")
        if apply_fn is None:
            raise RuntimeError(f"apply not defined in {self}. Did you forget to implement it?")
        return apply_fn

    @classmethod
    def can_edit(cls) -> bool:
        """
        Returns whether this tool can perform editing operations on code.

        :return: True if the tool can edit code, False otherwise
        """
        return issubclass(cls, ToolMarkerCanEdit)

    @classmethod
    def get_tool_description(cls) -> str:
        docstring = cls.__doc__
        if docstring is None:
            return ""
        return docstring.strip()

    @classmethod
    def get_apply_docstring_from_cls(cls) -> str:
        """Get the docstring for the apply method from the class (static metadata).
        Needed for creating MCP tools in a separate process without running into serialization issues.
        """
        # First try to get from __dict__ to handle dynamic docstring changes
        if "apply" in cls.__dict__:
            apply_fn = cls.__dict__["apply"]
        else:
            # Fall back to getattr for inherited methods
            apply_fn = getattr(cls, "apply", None)
            if apply_fn is None:
                raise AttributeError(f"apply method not defined in {cls}. Did you forget to implement it?")

        docstring = apply_fn.__doc__
        if not docstring:
            raise AttributeError(f"apply method has no (or empty) docstring in {cls}. Did you forget to implement it?")
        return docstring.strip()

    def get_apply_docstring(self) -> str:
        """Get the docstring for the apply method (instance method implementing ToolProtocol)."""
        return self.get_apply_docstring_from_cls()

    def get_apply_fn_metadata(self) -> FuncMetadata:
        """Get the metadata for the apply method (instance method implementing ToolProtocol)."""
        return self.get_apply_fn_metadata_from_cls()

    @classmethod
    def get_apply_fn_metadata_from_cls(cls) -> FuncMetadata:
        """Get the metadata for the apply method from the class (static metadata).
        Needed for creating MCP tools in a separate process without running into serialization issues.
        """
        # First try to get from __dict__ to handle dynamic docstring changes
        if "apply" in cls.__dict__:
            apply_fn = cls.__dict__["apply"]
        else:
            # Fall back to getattr for inherited methods
            apply_fn = getattr(cls, "apply", None)
            if apply_fn is None:
                raise AttributeError(f"apply method not defined in {cls}. Did you forget to implement it?")

        return func_metadata(apply_fn, skip_names=["self", "cls"])

    def _log_tool_application(self, frame: Any) -> None:
        params = {}
        ignored_params = {"self", "log_call", "catch_exceptions", "args", "apply_fn"}
        for param, value in frame.f_locals.items():
            if param in ignored_params:
                continue
            if param == "kwargs":
                params.update(value)
            else:
                params[param] = value
        log.info(f"{self.get_name_from_cls()}: {dict_string(params)}")

    @staticmethod
    def _limit_length(result: str, max_answer_chars: int) -> str:
        if (n_chars := len(result)) > max_answer_chars:
            result = (
                f"The answer is too long ({n_chars} characters). "
                + "Please try a more specific tool query or raise the max_answer_chars parameter."
            )
        return result

    def is_active(self) -> bool:
        return self.agent.tool_is_active(self.__class__)

    def apply_ex(self, log_call: bool = True, catch_exceptions: bool = True, **kwargs) -> str:  # type: ignore
        """
        Applies the tool with the given arguments
        """

        def task() -> str:
            apply_fn = self.get_apply_fn()

            try:
                if not self.is_active():
                    return f"Error: Tool '{self.get_name_from_cls()}' is not active. Active tools: {self.agent.get_active_tool_names()}"
            except Exception as e:
                return f"RuntimeError while checking if tool {self.get_name_from_cls()} is active: {e}"

            if log_call:
                self._log_tool_application(inspect.currentframe())
            try:
                # check whether the tool requires an active project and language server
                if not isinstance(self, ToolMarkerDoesNotRequireActiveProject):
                    if self.agent._active_project is None:
                        return (
                            "Error: No active project. Ask to user to select a project from this list: "
                            + f"{self.agent.serena_config.project_names}"
                        )
                    if not self.agent.is_language_server_running():
                        log.info("Language server is not running. Starting it ...")
                        self.agent.reset_language_server()

                # apply the actual tool
                result = apply_fn(**kwargs)

            except Exception as e:
                if not catch_exceptions:
                    raise
                msg = f"Error executing tool: {e}\n{traceback.format_exc()}"
                log.error(
                    f"Error executing tool: {e}. "
                    f"Consider restarting the language server to solve this (especially, if it's a timeout of a symbolic operation)",
                    exc_info=e,
                )
                result = msg

            if log_call:
                log.info(f"Result: {result}")

            try:
                self.language_server.save_cache()
            except Exception as e:
                log.error(f"Error saving language server cache: {e}")

            return result

        future = self.agent.issue_task(task, name=self.__class__.__name__)
        return future.result(timeout=self.agent.serena_config.tool_timeout)


class EditedFileContext:
    """
    Context manager for file editing.

    Create the context, then use `set_updated_content` to set the new content, the original content
    being provided in `original_content`.
    When exiting the context without an exception, the updated content will be written back to the file.
    """

    def __init__(self, relative_path: str, agent: "SerenaAgent"):
        self._project = agent.get_active_project()
        assert self._project is not None
        self._abs_path = os.path.join(self._project.project_root, relative_path)
        if not os.path.isfile(self._abs_path):
            raise FileNotFoundError(f"File {self._abs_path} does not exist.")
        with open(self._abs_path, encoding=self._project.project_config.encoding) as f:
            self._original_content = f.read()
        self._updated_content: str | None = None

    def __enter__(self) -> Self:
        return self

    def get_original_content(self) -> str:
        """
        :return: the original content of the file before any modifications.
        """
        return self._original_content

    def set_updated_content(self, content: str) -> None:
        """
        Sets the updated content of the file, which will be written back to the file
        when the context is exited without an exception.

        :param content: the updated content of the file
        """
        self._updated_content = content

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        if self._updated_content is not None and exc_type is None:
            assert self._project is not None
            with open(self._abs_path, "w", encoding=self._project.project_config.encoding) as f:
                f.write(self._updated_content)
            log.info(f"Updated content written to {self._abs_path}")
            # Language servers should automatically detect the change and update its state accordingly.
            # If they do not, we may have to add a call to notify it.


class CheckOnboardingPerformedTool(Tool):
    """
    Checks whether project onboarding was already performed.
    """

    def apply(self) -> str:
        """
        Checks whether project onboarding was already performed.
        You should always call this tool before beginning to actually work on the project/after activating a project,
        but after calling the initial instructions tool.
        """
        from .memory_tools import ListMemoriesTool

        list_memories_tool = self.agent.get_tool(ListMemoriesTool)
        memories = json.loads(list_memories_tool.apply())
        if len(memories) == 0:
            return (
                "Onboarding not performed yet (no memories available). "
                + "You should perform onboarding by calling the `onboarding` tool before proceeding with the task."
            )
        else:
            return f"""The onboarding was already performed, below is the list of available memories.
            Do not read them immediately, just remember that they exist and that you can read them later, if it is necessary
            for the current task.
            Some memories may be based on previous conversations, others may be general for the current project.
            You should be able to tell which one you need based on the name of the memory.
            
            {memories}"""


class OnboardingTool(Tool):
    """
    Performs onboarding (identifying the project structure and essential tasks, e.g. for testing or building).
    """

    def apply(self) -> str:
        """
        Call this tool if onboarding was not performed yet.
        You will call this tool at most once per conversation.

        :return: instructions on how to create the onboarding information
        """
        system = platform.system()
        return self.prompt_factory.create_onboarding_prompt(system=system)


class ThinkAboutCollectedInformationTool(Tool):
    """
    Thinking tool for pondering the completeness of collected information.
    """

    def apply(self) -> str:
        """
        Think about the collected information and whether it is sufficient and relevant.
        This tool should ALWAYS be called after you have completed a non-trivial sequence of searching steps like
        find_symbol, find_referencing_symbols, search_files_for_pattern, read_file, etc.
        """
        return self.prompt_factory.create_think_about_collected_information()


class ThinkAboutTaskAdherenceTool(Tool):
    """
    Thinking tool for determining whether the agent is still on track with the current task.
    """

    def apply(self) -> str:
        """
        Think about the task at hand and whether you are still on track.
        Especially important if the conversation has been going on for a while and there
        has been a lot of back and forth.

        This tool should ALWAYS be called before you insert, replace, or delete code.
        """
        return self.prompt_factory.create_think_about_task_adherence()


class ThinkAboutWhetherYouAreDoneTool(Tool):
    """
    Thinking tool for determining whether the task is truly completed.
    """

    def apply(self) -> str:
        """
        Whenever you feel that you are done with what the user has asked for, it is important to call this tool.
        """
        return self.prompt_factory.create_think_about_whether_you_are_done()


class SummarizeChangesTool(Tool):
    """
    Provides instructions for summarizing the changes made to the codebase.
    """

    def apply(self) -> str:
        """
        Summarize the changes you have made to the codebase.
        This tool should always be called after you have fully completed any non-trivial coding task,
        but only after the think_about_whether_you_are_done call.
        """
        return self.prompt_factory.create_summarize_changes()


class PrepareForNewConversationTool(Tool):
    """
    Provides instructions for preparing for a new conversation (in order to continue with the necessary context).
    """

    def apply(self) -> str:
        """
        Instructions for preparing for a new conversation. This tool should only be called on explicit user request.
        """
        return self.prompt_factory.create_prepare_for_new_conversation()


class InitialInstructionsTool(Tool, ToolMarkerDoesNotRequireActiveProject):
    """
    Gets the initial instructions for the current project.
    Should only be used in settings where the system prompt cannot be set,
    e.g. in clients you have no control over, like Claude Desktop.
    """

    def apply(self) -> str:
        """
        Get the initial instructions for the current coding project.
        You should always call this tool before starting to work (including using any other tool) on any programming task!
        The only exception is when a user asks you to activate a project, in which case you should call the `activate_project` first
        instead and then call this tool.
        """
        return self.agent.create_system_prompt()


class ToolRegistry:
    _tool_dict: dict[str, type[Tool]] | None = None
    """maps tool name to the corresponding tool class"""

    @staticmethod
    def _iter_tool_classes() -> Generator[type[Tool], None, None]:
        """
        Iterate over Tool subclasses.
        """
        yield from iter_subclasses(Tool)

    @classmethod
    def _get_tool_dict(cls) -> dict[str, type[Tool]]:
        if cls._tool_dict is None:
            cls._tool_dict = {}
            for tool_class in cls._iter_tool_classes():
                name = tool_class.get_name_from_cls()
                if name in cls._tool_dict:
                    raise ValueError(f"Duplicate tool name found: {name}. Tool classes must have unique names.")
                cls._tool_dict[name] = tool_class
        return cls._tool_dict

    @classmethod
    def get_tool_class_by_name(cls, tool_name: str) -> type[Tool]:
        try:
            return cls._get_tool_dict()[tool_name]
        except KeyError as e:
            available_tools = "\n".join(ToolRegistry.get_tool_names())
            raise ValueError(f"Tool with name {tool_name} not found. Available tools:\n{available_tools}") from e

    @classmethod
    def get_all_tool_classes(cls) -> list[type[Tool]]:
        return list(cls._get_tool_dict().values())

    @classmethod
    def get_tool_names(cls) -> list[str]:
        return list(cls._get_tool_dict().keys())

    @classmethod
    def tool_dict(cls) -> dict[str, type[Tool]]:
        """Maps tool name to the corresponding tool class"""
        return copy(cls._get_tool_dict())

    @classmethod
    def print_tool_overview(cls, tools: Iterable[type[Tool] | Tool] | None = None) -> None:
        """
        Print a summary of the tools. If no tools are passed, a summary of all tools is printed.
        """
        if tools is None:
            tools = cls._get_tool_dict().values()

        tool_dict: dict[str, type[Tool] | Tool] = {}
        for tool_class in tools:
            tool_dict[tool_class.get_name_from_cls()] = tool_class
        for tool_name in sorted(tool_dict.keys()):
            tool_class = tool_dict[tool_name]
            print(f" * `{tool_name}`: {tool_class.get_tool_description().strip()}")
