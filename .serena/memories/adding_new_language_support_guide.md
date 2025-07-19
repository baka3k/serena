# Adding New Language Support to Serena

This guide explains how to add support for a new programming language to Serena, using the C# implementation as a reference.

## Overview

Adding a new language involves:
1. **Language Server Implementation** - Creating a language-specific server class
2. **Language Registration** - Adding the language to enums and configurations  
3. **Test Repository** - Creating a minimal test project
4. **Test Suite** - Writing comprehensive tests
5. **Runtime Dependencies** - Configuring automatic language server downloads

## Step 1: Language Server Implementation

### 1.1 Create Language Server Class

Create a new file in `src/solidlsp/language_servers/` (e.g., `new_language_server.py`):

```python
from solidlsp.ls import SolidLanguageServer
from solidlsp.ls_config import LanguageServerConfig
from solidlsp.ls_logger import LanguageServerLogger
from solidlsp.lsp_protocol_handler.server import ProcessLaunchInfo

class NewLanguageServer(SolidLanguageServer):
    """
    Language server implementation for NewLanguage.
    """
    
    def __init__(self, config: LanguageServerConfig, logger: LanguageServerLogger, repository_root_path: str):
        # Determine language server command
        cmd = self._get_language_server_command()
        
        super().__init__(
            config,
            logger,
            repository_root_path,
            ProcessLaunchInfo(cmd=cmd, cwd=repository_root_path),
            "new_language",  # Language ID for LSP
        )
    
    def _get_language_server_command(self) -> list[str]:
        """Get the command to start the language server."""
        # Example: return ["new-language-server", "--stdio"]
        pass
    
    @override
    def is_ignored_dirname(self, dirname: str) -> bool:
        """Define language-specific directories to ignore."""
        return super().is_ignored_dirname(dirname) or dirname in ["build", "dist", "target"]
```

### 1.2 Language Server Discovery and Installation

For languages requiring automatic installation, implement download logic similar to C#:

```python
@classmethod
def _ensure_server_installed(cls, logger: LanguageServerLogger) -> str:
    """Ensure language server is installed and return path."""
    # Check system installation first
    system_server = shutil.which("new-language-server")
    if system_server:
        return system_server
    
    # Download and install if needed
    server_path = cls._download_and_install_server(logger)
    return server_path

def _download_and_install_server(cls, logger: LanguageServerLogger) -> str:
    """Download and install the language server."""
    # Implementation specific to your language server
    pass
```

### 1.3 LSP Initialization

Override initialization methods if needed:

```python
def _get_initialize_params(self) -> InitializeParams:
    """Return language-specific initialization parameters."""
    return {
        "processId": os.getpid(),
        "rootUri": PathUtils.path_to_uri(self.repository_root_path),
        "capabilities": {
            # Language-specific capabilities
        }
    }

def _start_server(self):
    """Start the language server with custom handlers."""
    # Set up notification handlers
    self.server.on_notification("window/logMessage", self._handle_log_message)
    
    # Start server and initialize
    self.server.start()
    init_response = self.server.send.initialize(self._get_initialize_params())
    self.server.notify.initialized({})
```

## Step 2: Language Registration

### 2.1 Add to Language Enum

In `src/solidlsp/ls_config.py`, add your language to the `Language` enum:

```python
class Language(str, Enum):
    # Existing languages...
    NEW_LANGUAGE = "new_language"
    
    def get_source_fn_matcher(self) -> FilenameMatcher:
        match self:
            # Existing cases...
            case self.NEW_LANGUAGE:
                return FilenameMatcher("*.newlang", "*.nl")  # File extensions
```

### 2.2 Update Language Server Factory

In `src/solidlsp/ls.py`, add your language to the `create` method:

```python
@classmethod
def create(cls, config: LanguageServerConfig, logger: LanguageServerLogger, repository_root_path: str) -> "SolidLanguageServer":
    match config.code_language:
        # Existing cases...
        case Language.NEW_LANGUAGE:
            from solidlsp.language_servers.new_language_server import NewLanguageServer
            return NewLanguageServer(config, logger, repository_root_path)
```

## Step 3: Test Repository

### 3.1 Create Test Project

Create a minimal project in `test/resources/repos/new_language/test_repo/`:

```
test/resources/repos/new_language/test_repo/
├── main.newlang              # Main source file
├── lib/
│   └── helper.newlang       # Additional source for testing
├── project.toml             # Project configuration (if applicable)
└── .gitignore              # Ignore build artifacts
```

### 3.2 Example Source Files

Create meaningful source files that demonstrate:
- **Classes/Types** - For symbol testing
- **Functions/Methods** - For reference finding
- **Imports/Dependencies** - For cross-file operations
- **Nested Structures** - For hierarchical symbol testing

Example `main.newlang`:
```
import lib.helper

class Calculator {
    func add(a: Int, b: Int) -> Int {
        return a + b
    }
    
    func subtract(a: Int, b: Int) -> Int {
        return helper.subtract(a, b)  // Reference to imported function
    }
}

class Program {
    func main() {
        let calc = Calculator()
        let result = calc.add(5, 3)  // Reference to add method
        print(result)
    }
}
```

## Step 4: Test Suite

### 4.1 Create Test File

Create `test/solidlsp/new_language/test_new_language_basic.py`:

```python
import os
import pytest

from solidlsp import SolidLanguageServer
from solidlsp.ls_config import Language
from solidlsp.ls_utils import SymbolUtils

@pytest.mark.new_language
class TestNewLanguageServer:
    @pytest.mark.parametrize("language_server", [Language.NEW_LANGUAGE], indirect=True)
    def test_find_symbol(self, language_server: SolidLanguageServer) -> None:
        """Test finding symbols in the full symbol tree."""
        symbols = language_server.request_full_symbol_tree()
        assert SymbolUtils.symbol_tree_contains_name(symbols, "Calculator"), "Calculator class not found"
        assert SymbolUtils.symbol_tree_contains_name(symbols, "add"), "add method not found"
        assert SymbolUtils.symbol_tree_contains_name(symbols, "Program"), "Program class not found"

    @pytest.mark.parametrize("language_server", [Language.NEW_LANGUAGE], indirect=True)
    def test_get_document_symbols(self, language_server: SolidLanguageServer) -> None:
        """Test getting document symbols."""
        file_path = "main.newlang"
        symbols = language_server.request_document_symbols(file_path)
        
        assert len(symbols) > 0
        
        # Check for expected classes
        class_names = [s.get("name") for s in symbols if s.get("kind") == 5]  # 5 = class
        assert "Calculator" in class_names
        assert "Program" in class_names

    @pytest.mark.parametrize("language_server", [Language.NEW_LANGUAGE], indirect=True)
    def test_find_references(self, language_server: SolidLanguageServer) -> None:
        """Test finding references to symbols."""
        file_path = "main.newlang"
        symbols = language_server.request_document_symbols(file_path)
        
        # Find the 'add' method
        add_symbol = None
        for sym in symbols:
            if sym.get("name") == "add":
                add_symbol = sym
                break
        
        assert add_symbol is not None, "Could not find 'add' method"
        
        # Test finding references
        sel_start = add_symbol["selectionRange"]["start"]
        refs = language_server.request_references(file_path, sel_start["line"], sel_start["character"])
        
        # Should find reference in main() method
        assert len(refs) > 0, "No references found for 'add' method"

    @pytest.mark.parametrize("language_server", [Language.NEW_LANGUAGE], indirect=True)
    def test_cross_file_references(self, language_server: SolidLanguageServer) -> None:
        """Test finding references across multiple files."""
        # Test imports and cross-file symbol usage
        file_path = "lib/helper.newlang"
        symbols = language_server.request_document_symbols(file_path)
        
        # Verify symbols exist in helper file
        assert len(symbols) > 0
        
        # Test that main.newlang references helper functions
        main_file = "main.newlang"
        main_symbols = language_server.request_document_symbols(main_file)
        assert len(main_symbols) > 0
```

### 4.2 Test Utilities and Edge Cases

Add additional test methods for:
- **Nested namespaces/modules**
- **Error handling** - Invalid syntax, missing files
- **Performance** - Large files, many symbols
- **Language-specific features** - Generics, interfaces, etc.

## Step 5: Runtime Dependencies (Optional)

If your language server needs automatic installation, add runtime dependencies to the language server class:

```python
from solidlsp.language_servers.common import RuntimeDependency

RUNTIME_DEPENDENCIES = [
    RuntimeDependency(
        id="NewLanguageServer",
        platform_id="linux-x64",
        package_name="new-language-server",
        package_version="1.0.0",
        url="https://github.com/new-lang/server/releases/download/v1.0.0/server-linux.tar.gz",
        archive_type="tar.gz",
        binary_name="new-language-server",
        extract_path="bin/"
    ),
    # Add other platforms as needed
]
```

## Step 6: Testing

### 6.1 Run Language-Specific Tests

```bash
# Run only your language tests
pytest -m new_language

# Run with verbose output
pytest -m new_language -v

# Run specific test file
pytest test/solidlsp/new_language/test_new_language_basic.py
```

### 6.2 Integration Testing

Test the full integration:

```python
# Test that language server can be created and started
from test.conftest import create_ls
from solidlsp.ls_config import Language

def test_integration():
    ls = create_ls(Language.NEW_LANGUAGE)
    ls.start()
    try:
        # Test basic operations
        symbols = ls.request_full_symbol_tree()
        assert len(symbols) > 0
    finally:
        ls.stop()
```

## Step 7: Documentation and Configuration

### 7.1 Update pytest.ini

Add your language marker to `pytest.ini`:

```ini
[tool.pytest.ini_options]
markers = [
    "new_language: marks tests as requiring NewLanguage language server"
]
```

### 7.2 CI/CD Configuration

Update GitHub Actions workflows to include your language in CI testing.

### 7.3 Documentation

Update:
- **README.md** - Add language to supported languages list
- **CHANGELOG.md** - Document the new language support
- **Language-specific docs** - Installation requirements, known issues

## Common Patterns and Best Practices

### Error Handling

```python
def _start_server(self):
    try:
        self.server.start()
        init_response = self.server.send.initialize(self._get_initialize_params())
    except Exception as e:
        raise LanguageServerException(f"Failed to start NewLanguage server: {e}") from e
```

### Performance Optimization

```python
@override
def is_ignored_dirname(self, dirname: str) -> bool:
    """Optimize by ignoring build/cache directories."""
    ignored_dirs = {"node_modules", "build", "dist", "target", ".cache"}
    return super().is_ignored_dirname(dirname) or dirname in ignored_dirs
```

### Language-Specific Configuration

```python
def _get_initialize_params(self) -> InitializeParams:
    """Configure for optimal NewLanguage support."""
    return {
        "processId": os.getpid(),
        "rootUri": PathUtils.path_to_uri(self.repository_root_path),
        "initializationOptions": {
            # Language-specific options
            "new_language": {
                "enable_formatting": True,
                "enable_diagnostics": True,
            }
        },
        "capabilities": {
            "textDocument": {
                "documentSymbol": {"hierarchicalDocumentSymbolSupport": True},
                "references": {"dynamicRegistration": True},
                "definition": {"dynamicRegistration": True},
            }
        }
    }
```

This comprehensive guide should enable you to add robust support for any new programming language to Serena, following the established patterns and ensuring thorough testing.