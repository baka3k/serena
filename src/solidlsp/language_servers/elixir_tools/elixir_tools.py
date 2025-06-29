import json
import logging
import os
import pathlib
import subprocess
import threading

from overrides import override

from solidlsp.ls import SolidLanguageServer
from solidlsp.ls_config import LanguageServerConfig
from solidlsp.ls_logger import LanguageServerLogger
from solidlsp.lsp_protocol_handler.lsp_types import InitializeParams
from solidlsp.lsp_protocol_handler.server import ProcessLaunchInfo


class ElixirTools(SolidLanguageServer):
    """
    Provides Elixir specific instantiation of the LanguageServer class using Next LS from elixir-tools.
    """

    @override
    def is_ignored_dirname(self, dirname: str) -> bool:
        # For Elixir projects, we should ignore:
        # - _build: compiled artifacts
        # - deps: dependencies
        # - node_modules: if the project has JavaScript components
        # - .elixir_ls: ElixirLS artifacts (in case both are present)
        # - cover: coverage reports
        return super().is_ignored_dirname(dirname) or dirname in ["_build", "deps", "node_modules", ".elixir_ls", "cover"]

    @staticmethod
    def _get_elixir_version():
        """Get the installed Elixir version or None if not found."""
        try:
            result = subprocess.run(["elixir", "--version"], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                return result.stdout.strip()
        except FileNotFoundError:
            return None
        return None

    @staticmethod
    def _get_nextls_version():
        """Get the installed Next LS version or None if not found."""
        try:
            result = subprocess.run(["nextls", "--version"], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                return result.stdout.strip()
        except FileNotFoundError:
            return None
        return None



    @classmethod
    def setup_runtime_dependency(cls):
        """
        Check if required Elixir runtime dependencies are available.
        Raises RuntimeError with helpful message if dependencies are missing.
        """
        elixir_version = cls._get_elixir_version()
        if not elixir_version:
            raise RuntimeError(
                "Elixir is not installed. Please install Elixir from https://elixir-lang.org/install.html and make sure it is added to your PATH."
            )

        nextls_version = cls._get_nextls_version()
        if not nextls_version:
            raise RuntimeError(
                "Found an Elixir version but Next LS is not installed.\n"
                "Please install Next LS from https://github.com/elixir-tools/next-ls#installation\n\n"
                "After installation, make sure it is added to your PATH."
            )

        return True

    def __init__(self, config: LanguageServerConfig, logger: LanguageServerLogger, repository_root_path: str):
        self.setup_runtime_dependency()

        super().__init__(
            config,
            logger,
            repository_root_path,
            ProcessLaunchInfo(cmd="nextls --stdio", cwd=repository_root_path),
            "elixir",
        )
        self.server_ready = threading.Event()
        self.request_id = 0

    def _get_initialize_params(self, repository_absolute_path: str) -> InitializeParams:
        """
        Returns the initialize params for the Next LS Language Server.
        """
        with open(os.path.join(os.path.dirname(__file__), "initialize_params.json"), encoding="utf-8") as f:
            d = json.load(f)

        del d["_description"]

        d["processId"] = os.getpid()
        assert d["rootPath"] == "$rootPath"
        d["rootPath"] = repository_absolute_path

        assert d["rootUri"] == "$rootUri"
        d["rootUri"] = pathlib.Path(repository_absolute_path).as_uri()

        assert d["workspaceFolders"][0]["uri"] == "$uri"
        d["workspaceFolders"][0]["uri"] = pathlib.Path(repository_absolute_path).as_uri()

        assert d["workspaceFolders"][0]["name"] == "$name"
        d["workspaceFolders"][0]["name"] = os.path.basename(repository_absolute_path)

        return d

    def _start_server(self):
        """Start Next LS server process"""

        def register_capability_handler(params):
            return

        def window_log_message(msg):
            self.logger.log(f"LSP: window/logMessage: {msg}", logging.INFO)

        def do_nothing(params):
            return

        def check_server_ready(params):
            # Next LS sends progress notifications when it's ready
            if params.get("value", {}).get("kind") == "end":
                self.server_ready.set()

        self.server.on_request("client/registerCapability", register_capability_handler)
        self.server.on_notification("window/logMessage", window_log_message)
        self.server.on_notification("$/progress", check_server_ready)
        self.server.on_notification("textDocument/publishDiagnostics", do_nothing)

        self.logger.log("Starting Next LS server process", logging.INFO)
        self.server.start()
        initialize_params = self._get_initialize_params(self.repository_root_path)

        self.logger.log(
            "Sending initialize request from LSP client to LSP server and awaiting response",
            logging.INFO,
        )
        init_response = self.server.send.initialize(initialize_params)

        # Verify server capabilities - be more lenient with Next LS
        self.logger.log(f"Next LS capabilities: {list(init_response['capabilities'].keys())}", logging.INFO)
        
        # Next LS may not provide all capabilities immediately, so we check for basic ones
        assert "textDocumentSync" in init_response["capabilities"], f"Missing textDocumentSync in {init_response['capabilities']}"
        
        # Some capabilities might be optional or provided later
        if "completionProvider" not in init_response["capabilities"]:
            self.logger.log("Warning: completionProvider not available in initial capabilities", logging.WARNING)
        if "definitionProvider" not in init_response["capabilities"]:
            self.logger.log("Warning: definitionProvider not available in initial capabilities", logging.WARNING)

        self.server.notify.initialized({})
        self.completions_available.set()

        # Next LS may take some time to be ready, so we wait for the progress notification
        self.server_ready.wait() 