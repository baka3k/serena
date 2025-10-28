import logging
from collections.abc import Callable, Iterator

from solidlsp import SolidLanguageServer
from solidlsp.ls_config import Language

log = logging.getLogger(__name__)


class LanguageServerManager:
    """
    Manages one or more language servers
    """

    def __init__(
        self,
        language_servers: dict[Language, SolidLanguageServer],
        language_server_factories: dict[Language, Callable[[], SolidLanguageServer]] | None = None,
    ) -> None:
        """
        :param language_servers: a mapping from language to language server; the servers are assumed to be already started.
            The first server in the iteration order is used as the default server.
            All servers are assumed to serve the same project root.
        :param language_server_factories: factories for the language servers, to be used for restarting them if needed
        """
        self._language_servers = language_servers
        self._language_server_factories = language_server_factories or {}
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

    def restart_language_server(self, language: Language) -> SolidLanguageServer:
        """
        Forces recreation and restart of the language server for the given language.

        :param language: the language
        :return: the newly created language server
        """
        factory = self._language_server_factories.get(language)
        if factory is None:
            raise ValueError(f"No language server factory found for language {language}; cannot recreate and restart")
        self._language_servers[language] = factory()
        self._language_servers[language].start()
        return self._language_servers[language]

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
