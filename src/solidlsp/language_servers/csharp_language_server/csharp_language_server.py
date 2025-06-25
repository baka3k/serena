"""
CSharp Language Server using csharp-ls (Roslyn-based LSP server)
"""

import logging
import os
import pathlib
import shutil
import subprocess
import threading

from overrides import override

from solidlsp.config import LanguageServerConfig
from solidlsp.exceptions import LanguageServerException
from solidlsp.logger import LanguageServerLogger
from solidlsp.ls import SolidLanguageServer
from solidlsp.ls_handler import ProcessLaunchInfo
from solidlsp.ls_types import InitializeParams


def breadth_first_file_scan(root_dir):
    """
    Perform a breadth-first scan of files in the given directory.
    Yields file paths in breadth-first order.
    """
    queue = [root_dir]
    while queue:
        current_dir = queue.pop(0)
        try:
            for item in os.listdir(current_dir):
                if item.startswith("."):
                    continue
                item_path = os.path.join(current_dir, item)
                if os.path.isdir(item_path):
                    queue.append(item_path)
                elif os.path.isfile(item_path):
                    yield item_path
        except (PermissionError, OSError):
            # Skip directories we can't access
            pass


def find_solution_or_project_file(root_dir) -> str | None:
    """
    Find the first .sln file in breadth-first order.
    If no .sln file is found, look for a .csproj file.
    """
    sln_file = None
    csproj_file = None
    
    for filename in breadth_first_file_scan(root_dir):
        if filename.endswith(".sln") and sln_file is None:
            sln_file = filename
        elif filename.endswith(".csproj") and csproj_file is None:
            csproj_file = filename
        
        # If we found a .sln file, return it immediately
        if sln_file:
            return sln_file
    
    # If no .sln file was found, return the first .csproj file
    return csproj_file


class CSharpLanguageServer(SolidLanguageServer):
    """
    Provides C# specific instantiation of the LanguageServer class using csharp-ls.
    csharp-ls is a Roslyn-based LSP server that provides modern C# language features.
    """

    def __init__(self, config: LanguageServerConfig, logger: LanguageServerLogger, repository_root_path: str):
        """
        Creates a CSharpLanguageServer instance. This class is not meant to be instantiated directly.
        Use LanguageServer.create() instead.
        """
        csharp_ls_path = self.setup_runtime_dependencies(logger, config)
        
        # Find solution or project file
        solution_or_project = find_solution_or_project_file(repository_root_path)
        
        # Build command
        cmd_parts = [csharp_ls_path]
        
        # Add logging level if debug is enabled
        if logger.logger.level <= logging.DEBUG:
            cmd_parts.extend(["--loglevel", "info"])
        else:
            cmd_parts.extend(["--loglevel", "error"])
        
        # Add solution file if found
        if solution_or_project:
            # Extract relative path from repository root
            rel_path = os.path.relpath(solution_or_project, repository_root_path)
            cmd_parts.extend(["--solution", rel_path])
            logger.log(f"Using solution/project file: {rel_path}", logging.INFO)
        else:
            logger.log("No .sln or .csproj file found, csharp-ls will attempt auto-discovery", logging.WARNING)
        
        cmd = " ".join(cmd_parts)
        
        super().__init__(
            config,
            logger,
            repository_root_path,
            ProcessLaunchInfo(cmd=cmd, cwd=repository_root_path),
            "csharp",
        )
        
        self.initialization_complete = threading.Event()

    @override
    def is_ignored_dirname(self, dirname: str) -> bool:
        return super().is_ignored_dirname(dirname) or dirname in ["bin", "obj", "packages", ".vs"]

    def setup_runtime_dependencies(self, logger: LanguageServerLogger, config: LanguageServerConfig) -> str:
        """
        Set up csharp-ls by ensuring it's installed as a dotnet tool.
        Returns the path to the csharp-ls executable.
        """
        # First check if csharp-ls is already available in PATH
        csharp_ls_path = shutil.which("csharp-ls")
        if csharp_ls_path:
            logger.log(f"Found csharp-ls in PATH: {csharp_ls_path}", logging.INFO)
            return csharp_ls_path
        
        # Check if dotnet is available
        dotnet_path = shutil.which("dotnet")
        if not dotnet_path:
            raise LanguageServerException(
                "dotnet SDK is not installed or not in PATH. "
                "Please install the .NET SDK from https://dotnet.microsoft.com/download"
            )
        
        # Check if csharp-ls is installed as a global tool
        try:
            result = subprocess.run(
                ["dotnet", "tool", "list", "-g"],
                capture_output=True,
                text=True,
                check=True
            )
            if "csharp-ls" in result.stdout:
                # csharp-ls is installed, but not in PATH
                # Try to find it in the default dotnet tools directory
                home = os.path.expanduser("~")
                possible_paths = [
                    os.path.join(home, ".dotnet", "tools", "csharp-ls"),
                    os.path.join(home, ".dotnet", "tools", "csharp-ls.exe"),
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        logger.log(f"Found csharp-ls at: {path}", logging.INFO)
                        return path
        except subprocess.CalledProcessError:
            pass
        
        # Install csharp-ls as a global tool
        logger.log("Installing csharp-ls as a global dotnet tool...", logging.INFO)
        try:
            subprocess.run(
                ["dotnet", "tool", "install", "-g", "csharp-ls"],
                check=True,
                capture_output=True,
                text=True
            )
            logger.log("Successfully installed csharp-ls", logging.INFO)
        except subprocess.CalledProcessError as e:
            # It might already be installed but failed to update
            if "is already installed" in e.stderr:
                logger.log("csharp-ls is already installed", logging.INFO)
            else:
                raise LanguageServerException(f"Failed to install csharp-ls: {e.stderr}")
        
        # After installation, try to find it again
        csharp_ls_path = shutil.which("csharp-ls")
        if csharp_ls_path:
            return csharp_ls_path
        
        # Try the default locations again
        home = os.path.expanduser("~")
        possible_paths = [
            os.path.join(home, ".dotnet", "tools", "csharp-ls"),
            os.path.join(home, ".dotnet", "tools", "csharp-ls.exe"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                logger.log(f"Found csharp-ls at: {path}", logging.INFO)
                return path
        
        raise LanguageServerException(
            "Failed to find csharp-ls after installation. "
            "Please ensure ~/.dotnet/tools is in your PATH"
        )

    def _get_initialize_params(self, repository_absolute_path: str) -> InitializeParams:
        """
        Returns the initialize params for the csharp-ls Language Server.
        """
        initialize_params: InitializeParams = {  # type: ignore
            "processId": os.getpid(),
            "rootPath": repository_absolute_path,
            "rootUri": pathlib.Path(repository_absolute_path).as_uri(),
            "capabilities": {
                "workspace": {
                    "applyEdit": True,
                    "workspaceEdit": {"documentChanges": True},
                    "didChangeConfiguration": {"dynamicRegistration": True},
                    "didChangeWatchedFiles": {"dynamicRegistration": True},
                    "symbol": {
                        "dynamicRegistration": True,
                        "symbolKind": {
                            "valueSet": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
                        },
                    },
                    "executeCommand": {"dynamicRegistration": True},
                },
                "textDocument": {
                    "synchronization": {
                        "dynamicRegistration": True,
                        "willSave": True,
                        "willSaveWaitUntil": True,
                        "didSave": True
                    },
                    "completion": {
                        "dynamicRegistration": True,
                        "contextSupport": True,
                        "completionItem": {
                            "snippetSupport": True,
                            "commitCharactersSupport": True,
                            "documentationFormat": ["markdown", "plaintext"],
                            "deprecatedSupport": True,
                            "preselectSupport": True,
                        },
                        "completionItemKind": {
                            "valueSet": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
                        },
                    },
                    "hover": {"dynamicRegistration": True, "contentFormat": ["markdown", "plaintext"]},
                    "signatureHelp": {
                        "dynamicRegistration": True,
                        "signatureInformation": {
                            "documentationFormat": ["markdown", "plaintext"],
                            "parameterInformation": {"labelOffsetSupport": True},
                        },
                    },
                    "definition": {"dynamicRegistration": True},
                    "references": {"dynamicRegistration": True},
                    "documentHighlight": {"dynamicRegistration": True},
                    "documentSymbol": {
                        "dynamicRegistration": True,
                        "symbolKind": {
                            "valueSet": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
                        },
                        "hierarchicalDocumentSymbolSupport": True,
                    },
                    "codeAction": {
                        "dynamicRegistration": True,
                        "codeActionLiteralSupport": {
                            "codeActionKind": {
                                "valueSet": [
                                    "",
                                    "quickfix",
                                    "refactor",
                                    "refactor.extract",
                                    "refactor.inline",
                                    "refactor.rewrite",
                                    "source",
                                    "source.organizeImports",
                                ]
                            }
                        },
                    },
                    "codeLens": {"dynamicRegistration": True},
                    "formatting": {"dynamicRegistration": True},
                    "rangeFormatting": {"dynamicRegistration": True},
                    "onTypeFormatting": {"dynamicRegistration": True},
                    "rename": {"dynamicRegistration": True},
                    "publishDiagnostics": {"relatedInformation": True},
                },
            },
            "workspaceFolders": [
                {"uri": pathlib.Path(repository_absolute_path).as_uri(), "name": os.path.basename(repository_absolute_path)}
            ],
        }

        return initialize_params

    def _start_server(self):
        """
        Starts the csharp-ls Language Server.
        
        Usage:
        ```
        async with lsp.start_server():
            # LanguageServer has been initialized
            await lsp.request_definition(...)
            await lsp.request_references(...)
            # Shutdown the LanguageServer on exit from scope
        # LanguageServer has been shutdown cleanly
        ```
        """
        
        def do_nothing(params):
            return

        def window_log_message(msg):
            """Log messages from the language server."""
            message_text = msg.get("message", "")
            self.logger.log(f"LSP: window/logMessage: {message_text}", logging.INFO)

        # Set up notification handlers
        self.server.on_notification("window/logMessage", window_log_message)
        self.server.on_notification("$/progress", do_nothing)
        self.server.on_notification("textDocument/publishDiagnostics", do_nothing)
        
        self.logger.log("Starting csharp-ls server process", logging.INFO)
        self.server.start()
        
        # Send initialization
        initialize_params = self._get_initialize_params(self.repository_root_path)
        
        self.logger.log("Sending initialize request to csharp-ls server", logging.INFO)
        init_response = self.server.send.initialize(initialize_params)
        self.logger.log(f"Received initialize response: {init_response}", logging.DEBUG)
        
        # Verify required capabilities
        capabilities = init_response.get("capabilities", {})
        assert "textDocumentSync" in capabilities
        assert "definitionProvider" in capabilities
        assert "referencesProvider" in capabilities
        assert "documentSymbolProvider" in capabilities
        
        # Complete initialization
        self.server.notify.initialized({})
        self.initialization_complete.set()
        self.completions_available.set()
        
        self.logger.log("csharp-ls server initialized and ready", logging.INFO)
