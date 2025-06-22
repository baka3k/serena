# Solid-LSP: Deadlock-Free Language Server Architecture

## Overview

Solid-LSP (`src/solidlsp/`) is a **simplified, deadlock-resistant** reimplementation of the language server communication layer, designed to replace the problematic `multilspy` architecture that suffered from asyncio contamination issues when used with MCP servers.

## The Problem Solid-LSP Solves

### Root Cause: Asyncio Context Contamination
The original `multilspy` implementation created **asyncio deadlocks** in MCP environments due to:

1. **Dual Event Loop Interference**: MCP server runs its own asyncio event loop, while SerenaAgent creates separate asyncio loops for language servers and dashboard
2. **Coroutine Leakage**: When MCP server calls SerenaAgent tools, coroutines created by `multilspy.LanguageServer.request_full_symbol_tree()` were leaked into the MCP server's asyncio context but never awaited
3. **Resource Exhaustion**: Accumulated unawaited coroutines in the MCP server's event loop eventually caused blocking and apparent "hangs"
4. **MCP-Specific Issue**: Only occurred when using MCP clients, not in direct script execution

### Evidence from Logs
```
INFO mcp.server.lowlevel.server:_handle_message:524 - Warning: RuntimeWarning: coroutine 'LanguageServer.request_full_symbol_tree' was never awaited
```

## Solid-LSP Architecture

### Core Design Principles

1. **Simplified Protocol Handling**: Eliminates the complex `lsp_protocol_handler` layer from `multilspy`
2. **Direct Async Management**: More explicit control over async operations without complex threading abstractions
3. **Single Process Operation**: Designed to work directly in the MCP server process without requiring process isolation
4. **Clean Asyncio Boundaries**: Prevents coroutine leakage between async contexts

### Key Architectural Differences from multilspy

#### 1. Simplified Handler Architecture
- **multilspy**: Complex `LanguageServerHandler` with extensive async task management and event loops
- **solid-lsp**: Streamlined `SolidLanguageServerHandler` with direct process management

#### 2. Protocol Simplification
- **No separate protocol handler package**: Direct LSP communication without abstraction layers
- **Reduced complexity**: Fewer components and dependencies
- **Same language server support**: Identical `language_servers/` implementations for all supported languages

#### 3. Enhanced Process Control
**solid-lsp** provides additional methods for finer process control:
- `_start_server_process()`: Direct process control
- `_start_server()`: Server initialization
- `_server_context` attribute: Enhanced server state management

## Process Isolation: No Longer Required

### Key Benefit: Single Process Operation

With solid-lsp, **process isolation is no longer needed** and is **disabled by default**:

```python
# src/serena/mcp.py:581-585
if USE_SOLID_LSP:
    mcp_factory = SerenaMCPFactorySingleProcess(context=context, project=project_file)
else:
    # using multilspy requires process isolation to prevent asyncio contamination
    mcp_factory = SerenaMCPFactoryWithProcessIsolation(context=context, project=project_file)
```

### Benefits of Single Process Operation
1. **Lower Resource Usage**: No separate processes for SerenaAgent and language servers
2. **Reduced Latency**: Direct method calls instead of inter-process communication
3. **Simplified Architecture**: No IPC overhead or process management complexity
4. **Better Performance**: Eliminated serialization/deserialization overhead
5. **Easier Debugging**: All components in same process, simplified stack traces

### Why multilspy Required Process Isolation
- **Asyncio Contamination**: multilspy's async patterns leaked coroutines into MCP server context
- **Mandatory Isolation**: Process boundaries were the only way to prevent deadlocks
- **Performance Penalty**: IPC overhead was necessary for stability

### How solid-lsp Enables Single Process
- **Clean Async Patterns**: No coroutine leakage between MCP server and language server contexts
- **Direct Integration**: Can safely run in same process as MCP server
- **Stable Operation**: No more timeout issues or hanging operations

## Supported Language Servers

Solid-LSP maintains compatibility with all language servers from `multilspy`:

### Primary Languages
- **Python**: Pyright Language Server (`src/solidlsp/language_servers/pyright_language_server/`)
- **Java**: Eclipse JDTLS (`src/solidlsp/language_servers/eclipse_jdtls/`)
- **TypeScript/JavaScript**: TypeScript Language Server (`src/solidlsp/language_servers/typescript_language_server/`)

### Additional Languages  
- **C#**: OmniSharp (`src/solidlsp/language_servers/omnisharp/`)
- **Rust**: Rust-Analyzer (`src/solidlsp/language_servers/rust_analyzer/`)
- **Go**: Gopls (`src/solidlsp/language_servers/gopls/`)
- **Ruby**: Solargraph (`src/solidlsp/language_servers/solargraph/`)
- **C++**: Clangd (`src/solidlsp/language_servers/clangd_language_server/`)
- **Dart**: Dart Language Server (`src/solidlsp/language_servers/dart_language_server/`)
- **PHP**: Intelephense (`src/solidlsp/language_servers/intelephense/`)
- **Kotlin**: Kotlin Language Server (`src/solidlsp/language_servers/kotlin_language_server/`)

## Implementation Details

### Core Classes

#### `SolidLanguageServer` (`src/solidlsp/ls.py`)
- **1,634 lines**: Main language server interface
- **Enhanced methods**: Same semantic capabilities as `multilspy.LanguageServer` but with improved async handling
- **Lifecycle control**: `start()`, `stop()`, `is_running()`, `language_server()` property (similar to multilspy's `SyncLanguageServer`)

#### `SolidLanguageServerHandler` (`src/solidlsp/ls_handler.py`)  
- **512 lines**: Simplified LSP protocol handler
- **Direct process management**: More straightforward than `multilspy`'s complex async task orchestration
- **Improved cleanup**: Better resource management and process termination

#### `SolidLspRequest` (`src/solidlsp/lsp_request.py`)
- **Simplified request handling**: Direct LSP request/response without complex protocol abstractions

### Configuration Integration

Solid-LSP is controlled by the `USE_SOLID_LSP` constant in `src/serena/constants.py`:

```python
USE_SOLID_LSP = True  # Default: enabled
```

This allows:
- **Default Operation**: solid-lsp enabled by default for stability
- **Fallback Capability**: Can switch back to multilspy + process isolation if needed
- **A/B Testing**: Easy comparison between implementations

## Deadlock Prevention

### How Solid-LSP Prevents Deadlocks

1. **Simplified Async Patterns**: Reduces complexity that led to coroutine leakage
2. **Clean Boundaries**: Proper async context management prevents MCP server contamination
3. **Direct Operations**: Eliminates abstraction layers where async context could be corrupted
4. **Single Process Safety**: Designed to work safely within MCP server process

### Verification

The solid-lsp approach provides:
- **No more timeout issues**: Clean async boundaries prevent MCP server contamination
- **Stable MCP operation**: Language server calls no longer hang after first timeout
- **Resource efficiency**: Lower memory usage without separate processes
- **Reliable semantic tools**: `find_symbol` and other tools work consistently
- **Better Performance**: Direct method calls instead of IPC

## Migration Strategy

### Current Status
- **Default**: Solid-LSP is the default implementation (`USE_SOLID_LSP = True`)
- **Stable**: Resolves known deadlock issues in MCP environments
- **Compatible**: Supports all existing language servers and semantic operations
- **Performant**: Single process operation with lower overhead

### Rollback Path
If issues arise, can fallback to multilspy with process isolation:
1. **Disable solid-lsp**: Set `USE_SOLID_LSP = False` 
2. **Automatic fallback**: System automatically uses multilspy + process isolation
3. **Performance trade-off**: Higher resource usage but guaranteed isolation

Solid-LSP represents an **architectural evolution** that eliminates the need for process isolation while maintaining all semantic capabilities and providing **better performance and stability** for MCP server deployments.