"""
The Serena Model Context Protocol (MCP) Server
"""

import os
import platform
import shutil
import sys
import threading
import webbrowser
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, TypeVar, Union

import click
import yaml
from pathspec import PathSpec
from ruamel.yaml.comments import CommentedMap
from sensai.util import logging
from sensai.util.logging import FallbackHandler, LogTime
from sensai.util.string import ToStringMixin

from serena import serena_version
from serena.config.context_mode import SerenaAgentContext, SerenaAgentMode
from serena.constants import (
    DEFAULT_ENCODING,
    PROJECT_TEMPLATE_FILE,
    REPO_ROOT,
    SELENA_CONFIG_TEMPLATE_FILE,
    SERENA_LOG_FORMAT,
    SERENA_MANAGED_DIR_NAME,
)
from serena.dashboard import MemoryLogHandler, SerenaDashboardAPI
from serena.prompt_factory import SerenaPromptFactory
from serena.symbol import SymbolManager
from serena.tools import Tool, ToolRegistry
from serena.util.file_system import GitignoreParser, match_path
from serena.util.general import load_yaml, save_yaml
from serena.util.inspection import determine_programming_language_composition
from solidlsp import SolidLanguageServer
from solidlsp.ls_config import Language, LanguageServerConfig
from solidlsp.ls_logger import LanguageServerLogger

if TYPE_CHECKING:
    from serena.gui_log_viewer import GuiLogViewerHandler

log = logging.getLogger(__name__)
TTool = TypeVar("TTool", bound="Tool")
T = TypeVar("T")
SUCCESS_RESULT = "OK"
DEFAULT_TOOL_TIMEOUT: float = 240


def show_fatal_exception_safe(e: Exception) -> None:
    """
    Shows the given exception in the GUI log viewer on the main thread and ensures that the exception is logged or at
    least printed to stderr.
    """
    # Make sure the error is logged (adding a fallback handler which writes to stderr in case there is no other handler)
    fallback_handler = FallbackHandler(logging.StreamHandler(sys.stderr))
    Logger.root.addHandler(fallback_handler)
    log.error(f"Fatal exception: {e}", exc_info=e)

    # attempt to show the error in the GUI
    try:
        # NOTE: The import can fail on macOS if Tk is not available (depends on Python interpreter installation, which uv
        #   used as a base); while tkinter as such is always available, its dependencies can be unavailable on macOS.
        from serena.gui_log_viewer import show_fatal_exception

        show_fatal_exception(e)
    except:
        pass


class SerenaConfigError(Exception):
    pass


def get_serena_managed_dir(project_root: str | Path) -> str:
    return os.path.join(project_root, SERENA_MANAGED_DIR_NAME)


def is_running_in_docker() -> bool:
    """Check if we're running inside a Docker container."""
    # Check for Docker-specific files
    if os.path.exists("/.dockerenv"):
        return True
    # Check cgroup for docker references
    try:
        with open("/proc/self/cgroup") as f:
            return "docker" in f.read()
    except FileNotFoundError:
        return False


@dataclass
class ProjectConfig(ToStringMixin):
    project_name: str
    language: Language
    ignored_paths: list[str] = field(default_factory=list)
    excluded_tools: set[str] = field(default_factory=set)
    read_only: bool = False
    ignore_all_files_in_gitignore: bool = True
    initial_prompt: str = ""
    encoding: str = DEFAULT_ENCODING

    SERENA_DEFAULT_PROJECT_FILE = "project.yml"

    @classmethod
    def autogenerate(cls, project_root: str | Path, project_name: str | None = None, save_to_disk: bool = True) -> Self:
        """
        Autogenerate a project configuration for a given project root.

        :param project_root: the path to the project root
        :param project_name: the name of the project; if None, the name of the project will be the name of the directory
            containing the project
        :param save_to_disk: whether to save the project configuration to disk
        :return: the project configuration
        """
        project_root = Path(project_root).resolve()
        if not project_root.exists():
            raise FileNotFoundError(f"Project root not found: {project_root}")
        project_name = project_name or project_root.name
        language_composition = determine_programming_language_composition(str(project_root))
        if len(language_composition) == 0:
            raise ValueError(
                f"Failed to autogenerate project.yaml: no programming language detected in project {project_root}. "
                f"You can either add some files that correspond to one of the supported programming languages, "
                f"or create the file {os.path.join(project_root, cls.rel_path_to_project_yml())} manually and specify the language there."
            )
        # find the language with the highest percentage
        dominant_language = max(language_composition.keys(), key=lambda lang: language_composition[lang])
        config_with_comments = load_yaml(PROJECT_TEMPLATE_FILE, preserve_comments=True)
        config_with_comments["project_name"] = project_name
        config_with_comments["language"] = dominant_language
        if save_to_disk:
            save_yaml(str(project_root / cls.rel_path_to_project_yml()), config_with_comments, preserve_comments=True)
        return cls._from_dict(config_with_comments)

    @classmethod
    def rel_path_to_project_yml(cls) -> str:
        return os.path.join(SERENA_MANAGED_DIR_NAME, cls.SERENA_DEFAULT_PROJECT_FILE)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Self:
        """
        Create a ProjectConfig instance from a configuration dictionary
        """
        language_str = data["language"].lower()
        project_name = data["project_name"]
        # backwards compatibility
        if language_str == "javascript":
            log.warning(f"Found deprecated project language `javascript` in project {project_name}, please change to `typescript`")
            language_str = "typescript"
        try:
            language = Language(language_str)
        except ValueError as e:
            raise ValueError(f"Invalid language: {data['language']}.\nValid languages are: {[l.value for l in Language]}") from e
        return cls(
            project_name=project_name,
            language=language,
            ignored_paths=data.get("ignored_paths", []),
            excluded_tools=set(data.get("excluded_tools", [])),
            read_only=data.get("read_only", False),
            ignore_all_files_in_gitignore=data.get("ignore_all_files_in_gitignore", True),
            initial_prompt=data.get("initial_prompt", ""),
            encoding=data.get("encoding", DEFAULT_ENCODING),
        )

    @classmethod
    def load(cls, project_root: Path | str, autogenerate: bool = True) -> Self:
        """
        Load a ProjectConfig instance from the path to the project root.
        """
        project_root = Path(project_root)
        yaml_path = project_root / cls.rel_path_to_project_yml()
        if not yaml_path.exists():
            if autogenerate:
                return cls.autogenerate(project_root)
            else:
                raise FileNotFoundError(f"Project configuration file not found: {yaml_path}")
        with open(yaml_path, encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
        if "project_name" not in yaml_data:
            yaml_data["project_name"] = project_root.name
        return cls._from_dict(yaml_data)

    def get_excluded_tool_classes(self) -> set[type["Tool"]]:
        return set(ToolRegistry.get_tool_class_by_name(tool_name) for tool_name in self.excluded_tools)


class ProjectNotFoundError(Exception):
    pass


@dataclass
class Project:
    project_root: str
    project_config: ProjectConfig

    @property
    def project_name(self) -> str:
        return self.project_config.project_name

    @property
    def language(self) -> Language:
        return self.project_config.language

    @classmethod
    def load(cls, project_root: str | Path, autogenerate: bool = True) -> Self:
        project_root = Path(project_root).resolve()
        if not project_root.exists():
            raise FileNotFoundError(f"Project root not found: {project_root}")
        project_config = ProjectConfig.load(project_root, autogenerate=autogenerate)
        return cls(project_root=str(project_root), project_config=project_config)

    def path_to_project_yml(self) -> str:
        return os.path.join(self.project_root, self.project_config.rel_path_to_project_yml())


@dataclass(kw_only=True)
class SerenaConfig:
    """
    Holds the Serena agent configuration, which is typically loaded from a YAML configuration file
    (when instantiated via :method:`from_config_file`), which is updated when projects are added or removed.
    For testing purposes, it can also be instantiated directly with the desired parameters.
    """

    projects: list[Project] = field(default_factory=list)
    gui_log_window_enabled: bool = False
    log_level: int = logging.INFO
    trace_lsp_communication: bool = False
    web_dashboard: bool = True
    web_dashboard_open_on_launch: bool = True
    tool_timeout: float = DEFAULT_TOOL_TIMEOUT
    loaded_commented_yaml: CommentedMap | None = None
    config_file_path: str | None = None
    """
    the path to the configuration file to which updates of the configuration shall be saved;
    if None, the configuration is not saved to disk
    """

    CONFIG_FILE = "serena_config.yml"
    CONFIG_FILE_DOCKER = "serena_config.docker.yml"  # Docker-specific config file; auto-generated if missing, mounted via docker-compose for user customization

    @classmethod
    def _generate_config_file(cls, config_file_path: str) -> None:
        """
        Generates a Serena configuration file at the specified path from the template file.

        :param config_file_path: the path where the configuration file should be generated
        """
        log.info(f"Auto-generating Serena configuration file in {config_file_path}")
        loaded_commented_yaml = load_yaml(SELENA_CONFIG_TEMPLATE_FILE, preserve_comments=True)
        save_yaml(config_file_path, loaded_commented_yaml, preserve_comments=True)

    @classmethod
    def _determine_config_file_path(cls) -> str:
        """
        :return: the location where the Serena configuration file is stored/should be stored
        """
        if is_running_in_docker():
            return os.path.join(REPO_ROOT, cls.CONFIG_FILE_DOCKER)
        else:
            candidates = [
                str(Path.home() / SERENA_MANAGED_DIR_NAME / cls.CONFIG_FILE),
                os.path.join(REPO_ROOT, cls.CONFIG_FILE),
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    return candidate
            return candidates[0]

    @classmethod
    def from_config_file(cls, generate_if_missing: bool = True) -> "SerenaConfig":
        """
        Static constructor to create SerenaConfig from the configuration file
        """
        config_file_path = cls._determine_config_file_path()

        # create the configuration file from the template if necessary
        if not os.path.exists(config_file_path):
            if not generate_if_missing:
                raise FileNotFoundError(f"Serena configuration file not found: {config_file_path}")
            log.info(f"Serena configuration file not found at {config_file_path}, autogenerating...")
            cls._generate_config_file(config_file_path)

        # load the configuration
        log.info(f"Loading Serena configuration from {config_file_path}")
        try:
            loaded_commented_yaml = load_yaml(config_file_path, preserve_comments=True)
        except Exception as e:
            raise ValueError(f"Error loading Serena configuration from {config_file_path}: {e}") from e

        # create the configuration instance
        instance = cls(loaded_commented_yaml=loaded_commented_yaml, config_file_path=config_file_path)

        # read projects
        if "projects" not in loaded_commented_yaml:
            raise SerenaConfigError("`projects` key not found in Serena configuration. Please update your `serena_config.yml` file.")

        # load list of known projects
        instance.projects = []
        num_project_migrations = 0
        for path in loaded_commented_yaml["projects"]:
            path = Path(path).resolve()
            if not path.exists() or (path.is_dir() and not (path / ProjectConfig.rel_path_to_project_yml()).exists()):
                log.warning(f"Project path {path} does not exist or does not contain a project configuration file, skipping.")
                continue
            if path.is_file():
                path = cls._migrate_out_of_project_config_file(path)
                if path is None:
                    continue
                num_project_migrations += 1
            project = Project.load(path)
            instance.projects.append(project)

        # set other configuration parameters
        if is_running_in_docker():
            instance.gui_log_window_enabled = False  # not supported in Docker
        else:
            instance.gui_log_window_enabled = loaded_commented_yaml.get("gui_log_window", False)
        instance.log_level = loaded_commented_yaml.get("log_level", loaded_commented_yaml.get("gui_log_level", logging.INFO))
        instance.web_dashboard = loaded_commented_yaml.get("web_dashboard", True)
        instance.web_dashboard_open_on_launch = loaded_commented_yaml.get("web_dashboard_open_on_launch", True)
        instance.tool_timeout = loaded_commented_yaml.get("tool_timeout", DEFAULT_TOOL_TIMEOUT)
        instance.trace_lsp_communication = loaded_commented_yaml.get("trace_lsp_communication", False)

        # re-save the configuration file if any migrations were performed
        if num_project_migrations > 0:
            log.info(
                f"Migrated {num_project_migrations} project configurations from legacy format to in-project configuration; re-saving configuration"
            )
            instance.save()

        return instance

    @classmethod
    def _migrate_out_of_project_config_file(cls, path: Path) -> Path | None:
        """
        Migrates a legacy project configuration file (which is a YAML file containing the project root) to the
        in-project configuration file (project.yml) inside the project root directory.

        :param path: the path to the legacy project configuration file
        :return: the project root path if the migration was successful, None otherwise.
        """
        log.info(f"Found legacy project configuration file {path}, migrating to in-project configuration.")
        try:
            with open(path, encoding="utf-8") as f:
                project_config_data = yaml.safe_load(f)
            if "project_name" not in project_config_data:
                project_name = path.stem
                with open(path, "a", encoding="utf-8") as f:
                    f.write(f"\nproject_name: {project_name}")
            project_root = project_config_data["project_root"]
            shutil.move(str(path), str(Path(project_root) / ProjectConfig.rel_path_to_project_yml()))
            return Path(project_root).resolve()
        except Exception as e:
            log.error(f"Error migrating configuration file: {e}")
            return None

    @cached_property
    def project_paths(self) -> list[str]:
        return sorted(project.project_root for project in self.projects)

    @cached_property
    def project_names(self) -> list[str]:
        return sorted(project.project_config.project_name for project in self.projects)

    def get_project(self, project_root_or_name: str) -> Project | None:
        for project in self.projects:
            if project.project_config.project_name == project_root_or_name:
                return project
        if os.path.isdir(project_root_or_name):
            project_root = Path(project_root_or_name).resolve()
            for project in self.projects:
                if Path(project.project_root).resolve() == project_root:
                    return project
        return None

    def add_project_from_path(self, project_root: Path | str, project_name: str | None = None) -> tuple[Project, bool]:
        """
        Add a project to the Serena configuration from a given path. Will raise a FileExistsError if the
        name or path is already registered.

        :param project_root: the path to the project to add
        :param project_name: the name of the project to add; if None, the name of the project will be the name of the directory
            containing the project
        :return: the project that was added and a boolean indicating whether a new project configuration was generated and
            saved to disk. It may be that no new project configuration was generated if the project configuration already
            exists on disk but the project itself was not added yet to the Serena configuration.
        """
        project_root = Path(project_root).resolve()
        if not project_root.exists():
            raise FileNotFoundError(f"Error: Path does not exist: {project_root}")
        if not project_root.is_dir():
            raise FileNotFoundError(f"Error: Path is not a directory: {project_root}")

        if project_name is None:
            project_name = project_root.name
        for already_registered_project in self.projects:
            if already_registered_project.project_name == project_name:
                raise FileExistsError(
                    f"Project name '{project_name}' already exists and points to {already_registered_project.project_root}."
                )
            if str(already_registered_project.project_root) == str(project_root):
                raise FileExistsError(
                    f"Project with path {project_root} was already added with name '{already_registered_project.project_name}'."
                )

        try:
            project_config = ProjectConfig.load(project_root)
            new_project_config_generated = False
        except FileNotFoundError:
            project_config = ProjectConfig.autogenerate(project_root, save_to_disk=True)
            new_project_config_generated = True

        new_project = Project(project_root=str(project_root), project_config=project_config)
        self.projects.append(new_project)
        self.save()

        return new_project, new_project_config_generated

    def remove_project(self, project_name: str) -> None:
        # find the index of the project with the desired name and remove it
        for i, project in enumerate(self.projects):
            if project.project_name == project_name:
                del self.projects[i]
                break
        else:
            raise ValueError(f"Project '{project_name}' not found in Serena configuration; valid project names: {self.project_names}")
        self.save()

    def save(self) -> None:
        """
        Saves the configuration to the file from which it was loaded (if any)
        """
        if self.config_file_path is None:
            return
        assert self.loaded_commented_yaml is not None, "Cannot save configuration without loaded YAML"
        loaded_original_yaml = deepcopy(self.loaded_commented_yaml)
        # projects are unique absolute paths
        # we also canonicalize them before saving
        loaded_original_yaml["projects"] = sorted({str(Path(project.project_root).resolve()) for project in self.projects})
        save_yaml(self.config_file_path, loaded_original_yaml, preserve_comments=True)


class LinesRead:
    def __init__(self) -> None:
        self.files: dict[str, set[tuple[int, int]]] = defaultdict(lambda: set())

    def add_lines_read(self, relative_path: str, lines: tuple[int, int]) -> None:
        self.files[relative_path].add(lines)

    def were_lines_read(self, relative_path: str, lines: tuple[int, int]) -> bool:
        lines_read_in_file = self.files[relative_path]
        return lines in lines_read_in_file

    def invalidate_lines_read(self, relative_path: str) -> None:
        if relative_path in self.files:
            del self.files[relative_path]


class MemoriesManager(ABC):
    @abstractmethod
    def load_memory(self, name: str) -> str:
        pass

    @abstractmethod
    def save_memory(self, name: str, content: str) -> str:
        pass

    @abstractmethod
    def list_memories(self) -> list[str]:
        pass

    @abstractmethod
    def delete_memory(self, name: str) -> str:
        pass


class MemoriesManagerMDFilesInProject(MemoriesManager):
    def __init__(self, project_root: str):
        self._memory_dir = Path(get_serena_managed_dir(project_root)) / "memories"
        self._memory_dir.mkdir(parents=True, exist_ok=True)

    def _get_memory_file_path(self, name: str) -> Path:
        # strip all .md from the name. Models tend to get confused, sometimes passing the .md extension and sometimes not.
        name = name.replace(".md", "")
        filename = f"{name}.md"
        return self._memory_dir / filename

    def load_memory(self, name: str) -> str:
        memory_file_path = self._get_memory_file_path(name)
        if not memory_file_path.exists():
            return f"Memory file {name} not found, consider creating it with the `write_memory` tool if you need it."
        with open(memory_file_path, encoding="utf-8") as f:
            return f.read()

    def save_memory(self, name: str, content: str) -> str:
        memory_file_path = self._get_memory_file_path(name)
        with open(memory_file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Memory {name} written."

    def list_memories(self) -> list[str]:
        return [f.name.replace(".md", "") for f in self._memory_dir.iterdir() if f.is_file()]

    def delete_memory(self, name: str) -> str:
        memory_file_path = self._get_memory_file_path(name)
        memory_file_path.unlink()
        return f"Memory {name} deleted."


def create_ls_for_project(
    project: str | Project,
    log_level: int = logging.INFO,
    ls_timeout: float | None = DEFAULT_TOOL_TIMEOUT - 5,
    trace_lsp_communication: bool = False,
) -> SolidLanguageServer:
    """
    Create a language server for a project. Note that you will have to start it
    before performing any LS operations.

    :param project: either a path to the project root or a ProjectConfig instance.
        If no project.yml is found, the default project configuration will be used.
    :param log_level: the log level for the language server
    :param ls_timeout: the timeout for the language server
    :param trace_lsp_communication: whether to trace LSP communication
    :return: the language server
    """
    if isinstance(project, str):
        project_instance = Project.load(project, autogenerate=True)
    else:
        project_instance = project

    project_config = project_instance.project_config
    ignored_paths = project_config.ignored_paths
    if len(ignored_paths) > 0:
        log.info(f"Using {len(ignored_paths)} ignored paths from the explicit project configuration.")
        log.debug(f"Ignored paths: {ignored_paths}")
    if project_config.ignore_all_files_in_gitignore:
        log.info(f"Parsing all gitignore files in {project_instance.project_root}")
        gitignore_parser = GitignoreParser(project_instance.project_root)
        log.info(f"Found {len(gitignore_parser.get_ignore_specs())} gitignore files.")
        for spec in gitignore_parser.get_ignore_specs():
            log.debug(f"Adding {len(spec.patterns)} patterns from {spec.file_path} to the ignored paths.")
            ignored_paths.extend(spec.patterns)
    log.debug(f"Using {len(ignored_paths)} ignored paths in total.")
    multilspy_config = LanguageServerConfig(
        code_language=project_instance.language,
        ignored_paths=ignored_paths,
        trace_lsp_communication=trace_lsp_communication,
    )
    ls_logger = LanguageServerLogger(log_level=log_level)
    log.info(f"Creating language server instance for {project_instance.project_root}.")
    return SolidLanguageServer.create(
        multilspy_config,
        ls_logger,
        project_instance.project_root,
        timeout=ls_timeout,
    )


@click.command()
@click.argument("project", type=click.Path(exists=True), required=False, default=os.getcwd())
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]), default="WARNING")
def index_project(project: str, log_level: str = "INFO") -> None:
    """
    Index a project by saving the symbols of files to Serena's language server cache.

    :param project: the project to index. By default, the current working directory is used.
    """
    log_level_int = logging.getLevelNamesMapping()[log_level.upper()]
    project = os.path.abspath(project)
    print(f"Indexing symbols in project {project}")
    ls = create_ls_for_project(project, log_level=log_level_int)
    with ls.start_server():
        ls.index_repository()
    print(f"Symbols saved to {ls.cache_path}")


class SerenaAgent:
    def __init__(
        self,
        project: str | None = None,
        project_activation_callback: Callable[[], None] | None = None,
        serena_config: SerenaConfig | None = None,
        context: SerenaAgentContext | None = None,
        modes: list[SerenaAgentMode] | None = None,
    ):
        """
        :param project: the project to load immediately or None to not load any project; may be a path to the project or a name of
            an already registered project;
        :param project_activation_callback: a callback function to be called when a project is activated.
        :param serena_config: the Serena configuration or None to read the configuration from the default location.
        :param context: the context in which the agent is operating, None for default context.
            The context may adjust prompts, tool availability, and tool descriptions.
        :param modes: list of modes in which the agent is operating (they will be combined), None for default modes.
            The modes may adjust prompts, tool availability, and tool descriptions.
        """
        # obtain serena configuration using the decoupled factory function
        self.serena_config = serena_config or SerenaConfig.from_config_file()

        # adjust log level
        serena_log_level = self.serena_config.log_level
        if Logger.root.level > serena_log_level:
            log.info(f"Changing the root logger level to {serena_log_level}")
            Logger.root.setLevel(serena_log_level)

        # open GUI log window if enabled
        self._gui_log_handler: Union["GuiLogViewerHandler", None] = None  # noqa
        if self.serena_config.gui_log_window_enabled:
            if platform.system() == "Darwin":
                log.warning("GUI log window is not supported on macOS")
            else:
                # even importing on macOS may fail if tkinter dependencies are unavailable (depends on Python interpreter installation
                # which uv used as a base, unfortunately)
                from serena.gui_log_viewer import GuiLogViewer, GuiLogViewerHandler

                self._gui_log_handler = GuiLogViewerHandler(
                    GuiLogViewer("dashboard", title="Serena Logs"), level=serena_log_level, format_string=SERENA_LOG_FORMAT
                )
                Logger.root.addHandler(self._gui_log_handler)

        # set the agent context
        if context is None:
            context = SerenaAgentContext.load_default()
        self._context = context

        # instantiate all tool classes
        self._all_tools: dict[type[Tool], Tool] = {tool_class: tool_class(self) for tool_class in ToolRegistry.get_all_tool_classes()}
        tool_names = [tool.get_name_from_cls() for tool in self._all_tools.values()]

        # determine the set exposed tools (which e.g. the MCP shall see), limited by the context
        # (which is fixed for the session)
        excluded_tool_classes = set(self._context.get_excluded_tool_classes())
        self._exposed_tools = {tc: t for tc, t in self._all_tools.items() if tc not in excluded_tool_classes}

        # If GUI log window is enabled, set the tool names for highlighting
        if self._gui_log_handler is not None:
            self._gui_log_handler.log_viewer.set_tool_names(tool_names)

        # start the dashboard (web frontend), registering its log handler
        if self.serena_config.web_dashboard:
            dashboard_log_handler = MemoryLogHandler(level=serena_log_level)
            Logger.root.addHandler(dashboard_log_handler)
            self._dashboard_thread, port = SerenaDashboardAPI(dashboard_log_handler, tool_names).run_in_thread()
            if self.serena_config.web_dashboard_open_on_launch:
                webbrowser.open(f"http://localhost:{port}/dashboard/index.html")

        # log fundamental information
        log.info(f"Starting Serena server (version={serena_version()}, process id={os.getpid()}, parent process id={os.getppid()})")
        log.info("Configuration file: %s", self.serena_config.config_file_path)
        log.info("Available projects: {}".format(", ".join(self.serena_config.project_names)))

        # create executor for starting the language server and running tools in another thread
        # This executor is used to achieve linear task execution, so it is important to use a single-threaded executor.
        self._task_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="SerenaAgentExecutor")
        self._task_executor_lock = threading.Lock()
        self._task_executor_task_index = 1

        # Initialize the prompt factory
        self.prompt_factory = SerenaPromptFactory()
        self._project_activation_callback = project_activation_callback

        # project-specific instances, which will be initialized upon project activation
        self._active_project: Project | None = None
        self._active_project_root: str | None = None
        self.language_server: SolidLanguageServer | None = None
        self.symbol_manager: SymbolManager | None = None
        self.memories_manager: MemoriesManager | None = None
        self.lines_read: LinesRead | None = None
        self.ignore_spec: PathSpec  # not set to None to avoid assert statements
        """Ignore spec, extracted from the project's gitignore files and the explicitly configured ignored paths."""

        # set the active modes
        if modes is None:
            modes = SerenaAgentMode.load_default_modes()
        self._modes = modes

        # log tool information
        log.info(f"Loaded tools ({len(self._all_tools)}): {', '.join([tool.get_name_from_cls() for tool in self._all_tools.values()])}")
        log.info(f"Number of exposed tools given {self._context}: {len(self._exposed_tools)}")

        self._active_tools: dict[type[Tool], Tool] = {}
        self._update_active_tools()

        # activate a project configuration (if provided or if there is only a single project available)
        if project is not None:
            try:
                self.activate_project_from_path_or_name(project)
            except ProjectNotFoundError as e:
                log.error(
                    f"Error activating project '{project}': {e}; Note that out-of-project configurations were migrated. "
                    "You should now pass either --project <project_name> or --project <project_root>."
                )

    def get_project_root(self) -> str:
        """
        :return: the root directory of the active project (if any); raises a ValueError if there is no active project
        """
        project = self.get_active_project()
        if project is None:
            raise ValueError("Cannot get project root if no project is active.")
        return project.project_root

    def path_is_inside_project(self, path: str | Path) -> bool:
        """
        Checks if the given (absolute or relative) path is inside the project directory.
        Note that even relative paths may be outside if the contain ".." or point to symlinks.
        """
        path = Path(path)
        _proj_root = Path(self.get_project_root())
        if not path.is_absolute():
            path = _proj_root / path

        path = path.resolve()
        return path.is_relative_to(_proj_root)

    def path_is_gitignored(self, path: str | Path) -> bool:
        """
        Checks if the given path is ignored by git. Non absolute paths are assumed to be relative to the project root.
        """
        path = Path(path)
        if path.is_absolute():
            relative_path = path.relative_to(self.get_project_root())
        else:
            relative_path = path

        # always ignore paths inside .git
        if len(relative_path.parts) > 0 and relative_path.parts[0] == ".git":
            return True

        return match_path(str(relative_path), self.ignore_spec, root_path=self.get_project_root())

    def validate_relative_path(self, relative_path: str) -> None:
        """
        Validates that the given relative path is safe to read or edit,
        meaning it's inside the project directory and is not ignored by git.
        """
        if not self.path_is_inside_project(relative_path):
            raise ValueError(f"{relative_path=} points to path outside of the repository root, can't use it for safety reasons")

        if self.path_is_gitignored(relative_path):
            raise ValueError(f"File {relative_path} is gitignored, can't read or edit it for safety reasons")

    def get_exposed_tool_instances(self) -> list["Tool"]:
        """
        :return: the tool instances which are exposed (e.g. to the MCP client).
            Note that the set of exposed tools is fixed for the session, as
            clients don't react to changes in the set of tools, so this is the superset
            of tools that can be offered during the session.
            If a client should attempt to use a tool that is dynamically disabled
            (e.g. because a project is activated that disables it), it will receive an error.
        """
        return list(self._exposed_tools.values())

    def get_active_project(self) -> Project | None:
        """
        :return: the active project or None if no project is active
        """
        return self._active_project

    def set_modes(self, modes: list[SerenaAgentMode]) -> None:
        """
        Set the current mode configurations.

        :param modes: List of mode names or paths to use
        """
        self._modes = modes
        self._update_active_tools()

        log.info(f"Set modes to {[mode.name for mode in modes]}")

    def get_active_modes(self) -> list[SerenaAgentMode]:
        """
        :return: the list of active modes
        """
        return list(self._modes)

    def create_system_prompt(self) -> str:
        return self.prompt_factory.create_system_prompt(
            context_system_prompt=self._context.prompt,
            mode_system_prompts=[mode.prompt for mode in self._modes],
        )

    def _update_active_tools(self) -> None:
        """
        Update the active tools based on context, modes, and project configuration.
        All tool exclusions are merged together.
        """
        excluded_tool_classes: set[type[Tool]] = set()
        # modes
        for mode in self._modes:
            mode_excluded_tool_classes = mode.get_excluded_tool_classes()
            if len(mode_excluded_tool_classes) > 0:
                log.info(
                    f"Mode {mode.name} excluded {len(mode_excluded_tool_classes)} tools: {', '.join([tool.get_name_from_cls() for tool in mode_excluded_tool_classes])}"
                )
                excluded_tool_classes.update(mode_excluded_tool_classes)
        # context
        context_excluded_tool_classes = self._context.get_excluded_tool_classes()
        if len(context_excluded_tool_classes) > 0:
            log.info(
                f"Context {self._context.name} excluded {len(context_excluded_tool_classes)} tools: {', '.join([tool.get_name_from_cls() for tool in context_excluded_tool_classes])}"
            )
            excluded_tool_classes.update(context_excluded_tool_classes)
        # project config
        if self._active_project is not None:
            project_excluded_tool_classes = self._active_project.project_config.get_excluded_tool_classes()
            if len(project_excluded_tool_classes) > 0:
                log.info(
                    f"Project {self._active_project.project_name} excluded {len(project_excluded_tool_classes)} tools: {', '.join([tool.get_name_from_cls() for tool in project_excluded_tool_classes])}"
                )
                excluded_tool_classes.update(project_excluded_tool_classes)
            if self._active_project.project_config.read_only:
                for tool_class in self._all_tools:
                    if tool_class.can_edit():
                        excluded_tool_classes.add(tool_class)

        self._active_tools = {
            tool_class: tool_instance for tool_class, tool_instance in self._all_tools.items() if tool_class not in excluded_tool_classes
        }

        log.info(f"Active tools after all exclusions ({len(self._active_tools)}): {', '.join(self.get_active_tool_names())}")

    def issue_task(self, task: Callable[[], Any], name: str | None = None) -> Future:
        """
        Issue a task to the executor for asynchronous execution.
        It is ensured that tasks are executed in the order they are issued, one after another.

        :param task: the task to execute
        :param name: the name of the task for logging purposes; if None, use the task function's name
        :return: a Future object representing the execution of the task
        """
        with self._task_executor_lock:
            task_name = f"Task-{self._task_executor_task_index}[{name or task.__name__}]"
            self._task_executor_task_index += 1

            def task_execution_wrapper() -> Any:
                with LogTime(task_name, logger=log):
                    return task()

            log.info(f"Scheduling {task_name}")
            return self._task_executor.submit(task_execution_wrapper)

    def execute_task(self, task: Callable[[], T]) -> T:
        """
        Executes the given task synchronously via the agent's task executor.
        This is useful for tasks that need to be executed immediately and whose results are needed right away.

        :param task: the task to execute
        :return: the result of the task execution
        """
        future = self.issue_task(task)
        return future.result()

    def _activate_project(self, project: Project) -> None:
        log.info(f"Activating {project.project_name} at {project.project_root}")
        self._active_project = project
        self._update_active_tools()

        # initialize project-specific instances which do not depend on the language server
        self.memories_manager = MemoriesManagerMDFilesInProject(project.project_root)
        self.lines_read = LinesRead()

        # reset project-specific instances that depend on the language server
        self.symbol_manager = None

        def init_language_server() -> None:
            # start the language server
            with LogTime("Language server initialization", logger=log):
                self.reset_language_server()
                assert self.language_server is not None
                self.ignore_spec = self.language_server.get_ignore_spec()

            # initialize project-specific instances which depend on the language server
            log.debug(f"Initializing symbol and memories manager for {project.project_name} at {project.project_root}")
            self.symbol_manager = SymbolManager(self.language_server, self)

        # initialize the language server in the background
        self.issue_task(init_language_server)

        if self._project_activation_callback is not None:
            self._project_activation_callback()

    def activate_project_from_path_or_name(self, project_root_or_name: str) -> tuple[Project, bool, bool]:
        """
        Activate a project from a path or a name.
        If the project was already registered, it will just be activated. If it was not registered,
        the project will be registered and activated. After that, the project can be activated again
        by name (not just by path).
        :return: a tuple of the project instance and two booleans indicating if a new project was added and if a new project configuration for the
            added project was generated.
        """
        new_project_generated = False
        new_project_config_generated = False
        project_instance: Project | None = self.serena_config.get_project(project_root_or_name)
        if project_instance is not None:
            log.info(f"Found registered project {project_instance.project_name} at path {project_instance.project_root}.")
        else:
            if not os.path.isdir(project_root_or_name):
                raise ProjectNotFoundError(
                    f"Project '{project_root_or_name}' not found: Not a valid project name or directory. "
                    f"Existing project names: {self.serena_config.project_names}"
                )
            project_instance, new_project_config_generated = self.serena_config.add_project_from_path(project_root_or_name)
            new_project_generated = True
            log.info(f"Added new project {project_instance.project_name} for path {project_instance.project_root}.")
            if new_project_config_generated:
                log.info(
                    f"Note: A new project configuration with language {project_instance.project_config.language.value} "
                    f"was autogenerated since no project configuration was found in {project_root_or_name}."
                )
        self._activate_project(project_instance)
        return project_instance, new_project_generated, new_project_config_generated

    def get_active_tool_classes(self) -> list[type["Tool"]]:
        """
        :return: the list of active tool classes for the current project
        """
        return list(self._active_tools.keys())

    def get_active_tool_names(self) -> list[str]:
        """
        :return: the list of names of the active tools for the current project
        """
        return sorted([tool.get_name_from_cls() for tool in self.get_active_tool_classes()])

    def tool_is_active(self, tool_class: type["Tool"] | str) -> bool:
        """
        :param tool_class: the class or name of the tool to check
        :return: True if the tool is active, False otherwise
        """
        if isinstance(tool_class, str):
            return tool_class in self.get_active_tool_names()
        else:
            return tool_class in self.get_active_tool_classes()

    def get_current_config_overview(self) -> str:
        """
        :return: a string overview of the current configuration, including the active and available configuration options
        """
        result_str = "Current configuration:\n"
        result_str += f"Serena version: {serena_version()}\n"
        result_str += f"Loglevel: {self.serena_config.log_level}, trace_lsp_communication={self.serena_config.trace_lsp_communication}\n"
        if self._active_project is not None:
            result_str += f"Active project: {self._active_project.project_name}\n"
        else:
            result_str += "No active project\n"
        result_str += "Available projects:\n" + "\n".join(list(self.serena_config.project_names)) + "\n"
        result_str += f"Active context: {self._context.name}\n"

        # Active modes
        active_mode_names = [mode.name for mode in self.get_active_modes()]
        result_str += "Active modes: {}\n".format(", ".join(active_mode_names)) + "\n"

        # Available but not active modes
        all_available_modes = SerenaAgentMode.list_registered_mode_names()
        inactive_modes = [mode for mode in all_available_modes if mode not in active_mode_names]
        if inactive_modes:
            result_str += "Available but not active modes: {}\n".format(", ".join(inactive_modes)) + "\n"

        # Active tools
        result_str += "Active tools (after all exclusions from the project, context, and modes):\n"
        active_tool_names = self.get_active_tool_names()
        # print the tool names in chunks
        chunk_size = 4
        for i in range(0, len(active_tool_names), chunk_size):
            chunk = active_tool_names[i : i + chunk_size]
            result_str += "  " + ", ".join(chunk) + "\n"

        # Available but not active tools
        all_tool_names = sorted([tool.get_name_from_cls() for tool in self._all_tools.values()])
        inactive_tool_names = [tool for tool in all_tool_names if tool not in active_tool_names]
        if inactive_tool_names:
            result_str += "Available but not active tools:\n"
            for i in range(0, len(inactive_tool_names), chunk_size):
                chunk = inactive_tool_names[i : i + chunk_size]
                result_str += "  " + ", ".join(chunk) + "\n"

        return result_str

    def is_language_server_running(self) -> bool:
        return self.language_server is not None and self.language_server.is_running()

    def reset_language_server(self) -> None:
        """
        Starts/resets the language server for the current project
        """
        tool_timeout = self.serena_config.tool_timeout
        if tool_timeout is None or tool_timeout < 0:
            ls_timeout = None
        else:
            if tool_timeout < 10:
                raise ValueError(f"Tool timeout must be at least 10 seconds, but is {tool_timeout} seconds")
            ls_timeout = tool_timeout - 5  # the LS timeout is for a single call, it should be smaller than the tool timeout

        # stop the language server if it is running
        if self.is_language_server_running():
            assert self.language_server is not None
            log.info(f"Stopping the current language server at {self.language_server.repository_root_path} ...")
            self.language_server.stop()
            self.language_server = None

        # instantiate and start the language server
        assert self._active_project is not None
        self.language_server = create_ls_for_project(
            self._active_project,
            log_level=self.serena_config.log_level,
            ls_timeout=ls_timeout,
            trace_lsp_communication=self.serena_config.trace_lsp_communication,
        )
        log.info(f"Starting the language server for {self._active_project.project_name}")
        self.language_server.start()
        if not self.language_server.is_running():
            raise RuntimeError(
                f"Failed to start the language server for {self._active_project.project_name} at {self._active_project.project_root}"
            )
        if self.symbol_manager is not None:
            log.debug("Setting the language server in the agent's symbol manager")
            self.symbol_manager.set_language_server(self.language_server)
        else:
            log.debug("No symbol manager available yet, skipping setting the language server")

    def get_tool(self, tool_class: type[TTool]) -> TTool:
        return self._all_tools[tool_class]  # type: ignore

    def print_tool_overview(self) -> None:
        ToolRegistry.print_tool_overview(self._active_tools.values())

    def mark_file_modified(self, relativ_path: str) -> None:
        assert self.lines_read is not None
        self.lines_read.invalidate_lines_read(relativ_path)

    def __del__(self) -> None:
        """
        Destructor to clean up the language server instance and GUI logger
        """
        if not hasattr(self, "_is_initialized"):
            return
        log.info("SerenaAgent is shutting down ...")
        if self.is_language_server_running():
            log.info("Stopping the language server ...")
            assert self.language_server is not None
            self.language_server.save_cache()
            self.language_server.stop()
        if self._gui_log_handler:
            log.info("Stopping the GUI log window ...")
            self._gui_log_handler.stop_viewer()
            Logger.root.removeHandler(self._gui_log_handler)
