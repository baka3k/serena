# Serena Project Structure Overview

## Top-Level Organization

Serena is organized into several key directories, each serving distinct architectural purposes:

```
serena/
├── src/                    # Main source code
│   ├── interprompt/        # Template and prompt management system
│   ├── serena/            # Core Serena functionality
│   └── solidlsp/          # Language server integration layer
├── tests/                 # Test suites
├── docs/                  # Documentation
├── prompts/               # Prompt templates and configurations
├── contexts/              # Context definitions (desktop-app, agent, ide-assistant)
├── modes/                 # Mode definitions (planning, editing, interactive, one-shot)
└── pyproject.toml         # Project configuration and dependencies
```

## Core Source Structure (`src/`)

### 1. **Serena Core (`src/serena/`)**
The heart of the Serena system containing:

#### **Agent and Core Logic**
- **`agent.py`**: Main `SerenaAgent` class, tool implementations, project management
- **`mcp.py`**: Model Context Protocol server implementation and MCP factories
- **`config.py`**: Configuration system (contexts, modes, registered configurations)
- **`constants.py`**: System constants including `USE_PROCESS_ISOLATION = False`

#### **Specialized Components**
- **`dashboard.py`**: Web dashboard for monitoring and logging (`MemoryLogHandler`, `SerenaDashboardAPI`)
- **`symbol.py`**: Symbol management (`SymbolLocation`, `Symbol`, `SymbolManager`)
- **`text_utils.py`**: Text processing utilities for search and file operations
- **`gui_log_viewer.py`**: GUI log window implementation

#### **Integration Layers**
- **`agno.py`**: Agno framework integration (`SerenaAgnoToolkit`, `SerenaAgnoAgentProvider`)
- **`process_isolated_agent.py`**: Process isolation infrastructure (available but unused by default)
- **`prompt_factory.py`**: Prompt generation and management

#### **Utility Modules (`src/serena/util/`)**
- **`file_system.py`**: File scanning, gitignore parsing (`GitignoreParser`, `scan_directory`)
- **`git.py`**: Git operations and status checking
- **`shell.py`**: Shell command execution (`execute_shell_command`)
- **`thread.py`**: Threading utilities with timeout support
- **`inspection.py`**: Code inspection and language detection utilities

### 2. **Solid-LSP (`src/solidlsp/`)**
Current language server integration layer:

#### **Core Language Server Components**
- **`ls.py`**: Main `SolidLanguageServer` class (1,634 lines)
- **`ls_handler.py`**: `SolidLanguageServerHandler` for protocol management (512 lines)
- **`ls_request.py`**: Request handling (`LanguageServerRequest`)
- **`ls_types.py`**: LSP type definitions
- **`ls_utils.py`**: Utility classes (`TextUtils`, `PathUtils`, `FileUtils`, `SymbolUtils`)

#### **Protocol Handling (`src/solidlsp/lsp_protocol_handler/`)**
- **`server.py`**: LSP protocol server implementation
- **`lsp_requests.py`**: LSP request and notification classes
- **`lsp_types.py`**: Complete LSP type system definitions
- **`lsp_constants.py`**: LSP protocol constants

#### **Language Server Implementations (`src/solidlsp/language_servers/`)**
Each language has its own subdirectory with specific implementations:
- **`pyright_language_server/`**: Python support via Pyright
- **`eclipse_jdtls/`**: Java support via Eclipse JDTLS
- **`typescript_language_server/`**: TypeScript/JavaScript support
- **`omnisharp/`**: C# support via OmniSharp
- **`rust_analyzer/`**: Rust support via Rust-Analyzer
- **`gopls/`**: Go support via Gopls
- **`solargraph/`**: Ruby support via Solargraph
- **`clangd_language_server/`**: C++ support via Clangd
- **`dart_language_server/`**: Dart support
- **`intelephense/`**: PHP support via Intelephense
- **`kotlin_language_server/`**: Kotlin support

### 3. **Interprompt (`src/interprompt/`)**
Template and prompt management system:

- **`multilang_prompt.py`**: Multi-language prompt templates (`MultiLangPromptTemplate`)
- **`jinja_template.py`**: Jinja2 template integration (`JinjaTemplate`)
- **`prompt_factory.py`**: Prompt factory base classes
- **`util/class_decorators.py`**: Utility decorators like `@singleton`

## Configuration and Templates

### **Prompt Templates (`prompts/`)**
Organized by context and functionality:
- **Context-specific prompts**: Different prompt sets for different execution contexts
- **Tool-specific prompts**: Specialized prompts for specific tools and operations
- **Multi-language support**: Prompts available in multiple languages when applicable

### **Contexts (`contexts/`)**
Execution environment definitions:
- **`desktop-app.yml`**: Desktop application context
- **`agent.yml`**: Agent-specific context
- **`ide-assistant.yml`**: IDE assistant context
- **Custom contexts**: Support for user-defined contexts

### **Modes (`modes/`)**
Behavior mode definitions:
- **`planning.yml`**: Planning mode behavior
- **`editing.yml`**: Editing mode behavior  
- **`interactive.yml`**: Interactive mode behavior
- **`one-shot.yml`**: One-shot execution mode

## Key Architectural Patterns

### 1. **Four-Layer Configuration Hierarchy**
1. **Global**: `serena_config.yml`
2. **CLI Arguments**: Runtime overrides
3. **Project**: `.serena/project.yml`
4. **Active Modes**: Runtime behavior modification

### 2. **Tool System Architecture**
- **Base Classes**: `ToolInterface`, `Tool` in `agent.py`
- **Marker Interfaces**: `ToolMarkerCanEdit`, `ToolMarkerDoesNotRequireActiveProject`
- **Tool Registry**: `ToolRegistry` with automatic tool discovery via `_iter_tool_classes()`
- **Tool Categories**: Semantic tools, file operations, project management, meta-tools

### 3. **Integration Patterns**
- **MCP Server**: Primary integration via `SerenaMCPFactory` classes
- **Agno Agent**: Model-agnostic integration via `SerenaAgnoToolkit`
- **Process Isolation**: Optional isolation via `ProcessIsolatedSerenaAgent`

### 4. **Memory Management**
- **Project Memories**: `.serena/memories/` directory for project-specific information
- **Memory Tools**: `WriteMemoryTool`, `ReadMemoryTool`, `ListMemoriesTool`
- **Memory Managers**: `MemoriesManager`, `MemoriesManagerMDFilesInProject`

## Dependencies and Build System

### **Development Tools**
- **`uv`**: Dependency management and virtual environment
- **`poe`**: Task orchestration (defined in `pyproject.toml`)
- **Essential Commands**: `uv run poe lint`, `uv run poe format`, `uv run poe type-check`, `uv run poe test`

### **Key Dependencies**
- **Language Server Protocol**: LSP implementations for each supported language
- **MCP (Model Context Protocol)**: For integration with Claude Desktop and other MCP clients
- **Agno Framework**: For model-agnostic agent implementations
- **AsyncIO**: For concurrent operations (carefully managed to avoid contamination)

## Project State Management

### **Project Configuration**
- **Global Config**: `serena_config.yml` in user home directory
- **Project Config**: `.serena/project.yml` in project root
- **Auto-generation**: Automatic creation of default configurations when missing

### **Language Detection**
- **Automatic**: Based on file composition analysis
- **Manual Override**: Via project configuration
- **Multi-language**: Support for projects with multiple languages

## Runtime Architecture

### **Default Operation Mode**
- **Single Process**: MCP server, agent, and language servers in same process
- **Direct Communication**: No IPC overhead
- **Solid-LSP**: Only language server implementation
- **Process Isolation**: Disabled by default (`USE_PROCESS_ISOLATION = False`)

### **Semantic Tool Integration**
- **Symbol-based Operations**: `find_symbol`, `replace_symbol_body`, `insert_after_symbol`
- **Regex-based Operations**: `replace_regex` for fine-grained edits
- **File Operations**: `read_file`, `create_text_file`, `list_dir`
- **Project Operations**: `search_for_pattern`, `get_symbols_overview`

This structure reflects Serena's evolution from a complex, multi-process system to a streamlined, single-process architecture that maintains all semantic capabilities while improving performance and reliability.