# Current Architecture: Solid-LSP Implementation

## Overview

Solid-LSP (`src/solidlsp/`) is Serena's current language server integration layer, designed as a **simplified, deadlock-resistant** replacement for the problematic multilspy architecture. It enables **single-process operation** with MCP servers while maintaining all semantic code analysis capabilities.

## Current System Status

- **Default Implementation**: Solid-LSP is the only language server implementation (multilspy completely removed)
- **Process Isolation**: **Disabled by default** (`USE_PROCESS_ISOLATION = False`)
- **MCP Integration**: Runs safely in the same process as MCP server
- **Performance**: Lower latency and resource usage than process-isolated multilspy

## Core Architecture

### Primary Components

#### 1. **`SolidLanguageServer`** (`src/solidlsp/ls.py`)
- **1,634 lines**: Main language server interface
- **Enhanced Methods**: Same semantic capabilities as multilspy but with improved async handling
- **Lifecycle Control**: `start()`, `stop()`, `is_running()`, `language_server()` property
- **Clean Async Patterns**: No coroutine leakage between async contexts

#### 2. **`SolidLanguageServerHandler`** (`src/solidlsp/ls_handler.py`)
- **512 lines**: Simplified LSP protocol handler
- **Direct Process Management**: More straightforward than multilspy's complex orchestration
- **Resource Management**: Better cleanup and process termination

#### 3. **Protocol Layer** (`src/solidlsp/lsp_protocol_handler/`)
- **Simplified Protocol**: Direct LSP communication without excessive abstraction
- **Types and Constants**: LSP type definitions and protocol constants
- **Request Handling**: Streamlined request/response pattern

### Language Server Support

Solid-LSP maintains **identical language support** to multilspy:

#### Primary Languages (Directly Supported)
- **Python**: Pyright Language Server (`src/solidlsp/language_servers/pyright_language_server/`)
- **Java**: Eclipse JDTLS (`src/solidlsp/language_servers/eclipse_jdtls/`)
- **TypeScript/JavaScript**: TypeScript Language Server (`src/solidlsp/language_servers/typescript_language_server/`)

#### Additional Languages (Full Support)  
- **C#**: OmniSharp (`src/solidlsp/language_servers/omnisharp/`)
- **Rust**: Rust-Analyzer (`src/solidlsp/language_servers/rust_analyzer/`)
- **Go**: Gopls (`src/solidlsp/language_servers/gopls/`)
- **Ruby**: Solargraph (`src/solidlsp/language_servers/solargraph/`)
- **C++**: Clangd (`src/solidlsp/language_servers/clangd_language_server/`)
- **Dart**: Dart Language Server (`src/solidlsp/language_servers/dart_language_server/`)
- **PHP**: Intelephense (`src/solidlsp/language_servers/intelephense/`)
- **Kotlin**: Kotlin Language Server (`src/solidlsp/language_servers/kotlin_language_server/`)

## Key Architectural Improvements

### 1. **Simplified Async Patterns**
- **Clean Boundaries**: Proper async context management prevents MCP server contamination
- **No Coroutine Leakage**: Eliminates the unawaited coroutine warnings from multilspy
- **Direct Operations**: Fewer abstraction layers where async context could be corrupted

### 2. **Single Process Safety**
- **Safe MCP Integration**: Designed to work within MCP server process without deadlocks
- **Eliminated IPC Overhead**: Direct method calls instead of inter-process communication
- **Resource Efficiency**: Lower memory usage and faster startup times

### 3. **Enhanced Process Control**
- **Direct Process Management**: `_start_server_process()`, `_start_server()` methods
- **Better State Management**: `_server_context` attribute for enhanced control
- **Improved Cleanup**: More reliable resource management and process termination

## Integration Points

### MCP Server Integration
Current implementation in `src/serena/mcp.py:590`:
```python
if not USE_PROCESS_ISOLATION:
    mcp_factory = SerenaMCPFactorySingleProcess(context=context, project=project_file)
else:
    mcp_factory = SerenaMCPFactoryWithProcessIsolation(context=context, project=project_file)
```

### Agent Integration
Language server creation in `src/serena/agent.py:642-689`:
```python
def create_ls_for_project(...) -> SolidLanguageServer:
    # Creates LanguageServerConfig with project settings
    # Returns SolidLanguageServer.create() instance
```

## Configuration and Settings

### Core Configuration
- **`USE_PROCESS_ISOLATION = False`**: Process isolation disabled by default
- **Language Detection**: Automatic based on project composition
- **Timeout Settings**: Configurable language server timeouts
- **LSP Communication Tracing**: Optional debugging support

### Project-Level Settings
- **Ignored Paths**: Respects project configuration and gitignore
- **Language Selection**: Automatic or manual language server selection  
- **Timeout Configuration**: Per-project timeout settings

## Performance Benefits

### vs. Multilspy + Process Isolation
1. **Lower Latency**: Direct method calls instead of IPC
2. **Reduced Memory**: Single process instead of multiple processes
3. **Faster Startup**: No process creation overhead
4. **Simpler Debugging**: All components in same process with unified stack traces

### Stability Improvements
1. **No Asyncio Deadlocks**: Clean async boundaries prevent contamination
2. **Reliable Operations**: `find_symbol` and other tools work consistently
3. **Resource Management**: Better cleanup prevents resource leaks
4. **Error Handling**: Simplified error propagation and handling

## Current Operational State

- **Production Ready**: Default implementation for all Serena deployments
- **Fully Tested**: All semantic tools (`find_symbol`, `replace_symbol_body`, etc.) working reliably
- **MCP Compatible**: Stable operation with Claude Desktop and other MCP clients
- **Cross-Platform**: Works on all supported operating systems

This architecture represents the successful resolution of the multilspy asyncio contamination issues while maintaining full semantic code analysis capabilities and improving overall performance.