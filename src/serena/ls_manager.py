import logging
from collections.abc import Iterator
from dataclasses import dataclass

from serena.constants import SERENA_MANAGED_DIR_IN_HOME, SERENA_MANAGED_DIR_NAME
from solidlsp import SolidLanguageServer
from solidlsp.ls_config import Language, LanguageServerConfig
from solidlsp.ls_logger import LanguageServerLogger
from solidlsp.settings import SolidLSPSettings

log = logging.getLogger(__name__)


@dataclass
class LanguageServerFactoryParams:
    project_root: str
    encoding: str
    ignored_patterns: list[str]
    ls_timeout: float | None = None
    ls_specific_settings: dict | None = None
    log_level: int = logging.INFO
    trace_lsp_communication: bool = False


class LanguageServerFactory:
    def __init__(self, language: Language):
        self.language = language

    def create_language_server(self, params: LanguageServerFactoryParams) -> SolidLanguageServer:
        ls_config = LanguageServerConfig(
            code_language=self.language,
            ignored_paths=params.ignored_patterns,
            trace_lsp_communication=params.trace_lsp_communication,
            encoding=params.encoding,
        )
        ls_logger = LanguageServerLogger(log_level=params.log_level)

        log.info(f"Creating language server instance for {params.project_root}, language={self.language}.")
        return SolidLanguageServer.create(
            ls_config,
            ls_logger,
            params.project_root,
            timeout=params.ls_timeout,
            solidlsp_settings=SolidLSPSettings(
                solidlsp_dir=SERENA_MANAGED_DIR_IN_HOME,
                project_data_relative_path=SERENA_MANAGED_DIR_NAME,
                ls_specific_settings=params.ls_specific_settings or {},
            ),
        )


class LanguageServerManager:
    """
    Manages one or more language servers
    """

    def __init__(
        self,
        language_servers: dict[Language, SolidLanguageServer],
        language_server_factories: dict[Language, LanguageServerFactory] | None = None,
        language_server_factory_params: LanguageServerFactoryParams | None = None,
    ) -> None:
        """
        :param language_servers: a mapping from language to language server; the servers are assumed to be already started.
            The first server in the iteration order is used as the default server.
            All servers are assumed to serve the same project root.
        :param language_server_factories: factories for the language servers, to be used for restarting them if needed
        :param language_server_factory_params: parameters to be used when (re-)creating language servers via factories
        """
        self._language_servers = language_servers
        self._language_server_factories = language_server_factories or {}
        self._language_server_factory_params = language_server_factory_params
        self._default_language_server = next(iter(language_servers.values()))
        self._root_path = self._default_language_server.repository_root_path

    def get_root_path(self) -> str:
        return self._root_path

    def _ensure_functional_ls(self, ls: SolidLanguageServer) -> SolidLanguageServer:
        if not ls.is_running():
            log.warning(f"Language server for language {ls.language} is not running; restarting ...")
            ls = self.restart_language_server(ls.language)
        return ls

    def get_language_server(self, relative_path: str) -> SolidLanguageServer:
        ls: SolidLanguageServer | None = None
        if len(self._language_servers) > 1:
            for candidate in self._language_servers.values():
                if not candidate.is_ignored_path(relative_path, ignore_unsupported_files=True):
                    ls = candidate
                    break
        if ls is None:
            ls = self._default_language_server
        return self._ensure_functional_ls(ls)

    def _create_and_start_language_server(self, language: Language) -> SolidLanguageServer:
        factory = self._language_server_factories.get(language)
        if factory is None:
            raise ValueError(f"No language server factory found for language {language}; cannot create new instance")
        if self._language_server_factory_params is None:
            raise ValueError("No language server factory parameters provided; cannot instantiate new language server instance")
        self._language_servers[language] = factory.create_language_server(self._language_server_factory_params)
        self._language_servers[language].start()
        return self._language_servers[language]

    def restart_language_server(self, language: Language) -> SolidLanguageServer:
        """
        Forces recreation and restart of the language server for the given language.
        It is assumed that the language server for the given language is no longer running.

        :param language: the language
        :return: the newly created language server
        """
        return self._create_and_start_language_server(language)

    def add_language_server(self, language: Language) -> SolidLanguageServer:
        """
        Dynamically adds a new language server for the given language.

        :param language: the language
        :param factory: the factory to create the language server
        :return: the newly created language server
        """
        if language in self._language_servers:
            raise ValueError(f"Language server for language {language} already present")
        self._language_server_factories[language] = LanguageServerFactory(language)
        try:
            return self._create_and_start_language_server(language)
        except Exception as e:
            del self._language_server_factories[language]
            if language in self._language_servers:
                del self._language_servers[language]
            raise Exception(f"Addition of language server for {language.value} failed") from e

    def iter_language_servers(self) -> Iterator[SolidLanguageServer]:
        for ls in self._language_servers.values():
            yield self._ensure_functional_ls(ls)

    def stop_all(self, save_cache: bool = False) -> None:
        """
        Stops all managed language servers.

        :param save_cache: whether to save the cache before stopping
        """
        for ls in self.iter_language_servers():
            if ls.is_running():
                if save_cache:
                    ls.save_cache()
                log.info(f"Stopping language server for language {ls.language} ...")
                ls.stop()

    def save_all_caches(self) -> None:
        """
        Saves the caches of all managed language servers.
        """
        for ls in self.iter_language_servers():
            if ls.is_running():
                ls.save_cache()
