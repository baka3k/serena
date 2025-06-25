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

from solidlsp.ls_config import LanguageServerConfig
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
    Provides C# specific instantiation of the LanguageServer class using Microsoft.CodeAnalysis.LanguageServer.
    This is the official Roslyn-based language server from Microsoft.
    """

    def __init__(self, config: LanguageServerConfig, logger: LanguageServerLogger, repository_root_path: str):
        """
        Creates a CSharpLanguageServer instance. This class is not meant to be instantiated directly.
        Use LanguageServer.create() instead.
        """
        language_server_path = self.setup_runtime_dependencies(logger, config)
        
        # Find solution or project file
        solution_or_project = find_solution_or_project_file(repository_root_path)
        
        # Build command - Microsoft.CodeAnalysis.LanguageServer uses stdio by default
        cmd_parts = ["dotnet", language_server_path]
        
        # Add logging level if debug is enabled
        if logger.logger.level <= logging.DEBUG:
            cmd_parts.extend(["--logLevel", "Information"])
        
        # The language server will discover the solution/project from the workspace root
        if solution_or_project:
            logger.log(f"Found solution/project file: {solution_or_project}", logging.INFO)
        else:
            logger.log("No .sln or .csproj file found, language server will attempt auto-discovery", logging.WARNING)
        
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
        Set up Microsoft.CodeAnalysis.LanguageServer by downloading the NuGet package.
        Returns the path to the language server DLL.
        """
        import platform
        import tempfile
        from pathlib import Path
        
        # Determine the runtime ID based on the platform
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        # Map platform info to runtime ID
        if system == "windows":
            runtime_id = "win-x64" if machine in ["amd64", "x86_64"] else "win-arm64"
        elif system == "darwin":
            runtime_id = "osx-x64" if machine in ["x86_64"] else "osx-arm64"
        elif system == "linux":
            # Check if we're on musl or glibc
            # For now, assume glibc (most common)
            runtime_id = "linux-x64" if machine in ["x86_64", "amd64"] else "linux-arm64"
        else:
            # Fallback to neutral package
            runtime_id = "neutral"
        
        # Package configuration
        package_name = f"Microsoft.CodeAnalysis.LanguageServer.{runtime_id}"
        package_version = "4.13.0-2.final"  # Latest stable version as of search
        
        # Check if already downloaded
        cache_dir = Path.home() / ".cache" / "serena" / "language-servers" / "csharp"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        server_dir = cache_dir / f"{package_name}.{package_version}"
        server_dll = server_dir / "Microsoft.CodeAnalysis.LanguageServer.dll"
        
        if server_dll.exists():
            logger.log(f"Using cached Microsoft.CodeAnalysis.LanguageServer from {server_dll}", logging.INFO)
            return str(server_dll)
        
        # Download the package
        logger.log(f"Downloading {package_name} version {package_version}...", logging.INFO)
        
        # Check if nuget or dotnet is available
        nuget_cmd = shutil.which("nuget")
        dotnet_cmd = shutil.which("dotnet")
        
        if not nuget_cmd and not dotnet_cmd:
            raise LanguageServerException(
                "Neither nuget nor dotnet CLI is available. "
                "Please install .NET SDK from https://dotnet.microsoft.com/download"
            )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            if dotnet_cmd:
                # Use dotnet restore to download the package
                project_content = f"""<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="{package_name}" Version="{package_version}" />
  </ItemGroup>
</Project>"""
                
                project_file = temp_path / "temp.csproj"
                project_file.write_text(project_content)
                
                try:
                    # Restore the package
                    subprocess.run(
                        [dotnet_cmd, "restore", str(project_file), "--packages", str(temp_path)],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    
                    # Find the downloaded package
                    package_path = temp_path / package_name.lower() / package_version
                    
                except subprocess.CalledProcessError as e:
                    raise LanguageServerException(f"Failed to download package: {e.stderr}")
            
            else:
                # Use nuget to download the package
                try:
                    subprocess.run(
                        [
                            nuget_cmd, "install", package_name,
                            "-Version", package_version,
                            "-OutputDirectory", str(temp_path),
                            "-NonInteractive"
                        ],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    
                    # Find the downloaded package
                    package_path = temp_path / f"{package_name}.{package_version}"
                    
                except subprocess.CalledProcessError as e:
                    raise LanguageServerException(f"Failed to download package: {e.stderr}")
            
            # Extract the language server files
            if runtime_id == "neutral":
                # For neutral package, files are in lib/net8.0
                source_dir = package_path / "lib" / "net8.0"
            else:
                # For runtime-specific packages, files are in content/LanguageServer/{runtime-id}
                source_dir = package_path / "content" / "LanguageServer" / runtime_id
            
            if not source_dir.exists():
                # Try alternative locations
                for possible_dir in [
                    package_path / "tools" / "net8.0" / "any",
                    package_path / "lib" / "net8.0",
                    package_path / "contentFiles" / "any" / "net8.0"
                ]:
                    if possible_dir.exists():
                        source_dir = possible_dir
                        break
                else:
                    raise LanguageServerException(
                        f"Could not find language server files in package. "
                        f"Searched in {package_path}"
                    )
            
            # Copy files to cache directory
            server_dir.mkdir(parents=True, exist_ok=True)
            
            import shutil as shutil_module
            shutil_module.copytree(source_dir, server_dir, dirs_exist_ok=True)
            
            if not server_dll.exists():
                raise LanguageServerException(
                    "Microsoft.CodeAnalysis.LanguageServer.dll not found after extraction"
                )
            
            # Make the DLL executable on Unix-like systems
            if system != "windows":
                import stat
                server_dll.chmod(server_dll.stat().st_mode | stat.S_IEXEC)
            
            logger.log(f"Successfully installed Microsoft.CodeAnalysis.LanguageServer to {server_dll}", logging.INFO)
            return str(server_dll)

    def _get_initialize_params(self, repository_absolute_path: str) -> InitializeParams:
        """
        Returns the initialize params for the Microsoft.CodeAnalysis.LanguageServer.
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
                    "configuration": True,
                    "workspaceFolders": True,
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
                    "foldingRange": {
                        "dynamicRegistration": True,
                        "rangeLimit": 5000,
                        "lineFoldingOnly": True
                    },
                },
            },
            "workspaceFolders": [
                {"uri": pathlib.Path(repository_absolute_path).as_uri(), "name": os.path.basename(repository_absolute_path)}
            ],
        }

        return initialize_params

    def _start_server(self):
        """
        Starts the Microsoft.CodeAnalysis.LanguageServer.
        
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
            level = msg.get("type", 4)  # Default to Log level
            
            # Map LSP message types to Python logging levels
            level_map = {
                1: logging.ERROR,    # Error
                2: logging.WARNING,  # Warning
                3: logging.INFO,     # Info
                4: logging.DEBUG     # Log
            }
            
            self.logger.log(f"LSP: {message_text}", level_map.get(level, logging.DEBUG))

        def handle_workspace_configuration(params):
            """Handle workspace/configuration requests from the server."""
            # Return empty configuration for now
            items = params.get("items", [])
            return [{}] * len(items)

        # Set up notification handlers
        self.server.on_notification("window/logMessage", window_log_message)
        self.server.on_notification("$/progress", do_nothing)
        self.server.on_notification("textDocument/publishDiagnostics", do_nothing)
        self.server.on_request("workspace/configuration", handle_workspace_configuration)
        
        self.logger.log("Starting Microsoft.CodeAnalysis.LanguageServer process", logging.INFO)
        self.server.start()
        
        # Send initialization
        initialize_params = self._get_initialize_params(self.repository_root_path)
        
        self.logger.log("Sending initialize request to language server", logging.INFO)
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
        
        self.logger.log("Microsoft.CodeAnalysis.LanguageServer initialized and ready", logging.INFO)
