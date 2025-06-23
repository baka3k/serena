# Supported Language Servers in Solid-LSP

Serena currently supports multiple programming languages through language servers via the solid-lsp architecture located in `src/solidlsp/language_servers/`.

## Primary Languages (Directly Supported)

### Python
- **Language Server**: Pyright Language Server
- **Location**: `src/solidlsp/language_servers/pyright_language_server/`
- **Features**: Full semantic analysis, type checking, symbol navigation
- **Previously**: Also supported Jedi (now deprecated in favor of Pyright)

### Java  
- **Language Server**: Eclipse JDTLS (Java Development Tools Language Server)
- **Location**: `src/solidlsp/language_servers/eclipse_jdtls/`
- **Features**: Full Java language support, Maven/Gradle integration
- **Note**: Startup can be slow, especially on initial launch

### TypeScript/JavaScript
- **Language Server**: TypeScript Language Server
- **Location**: `src/solidlsp/language_servers/typescript_language_server/`
- **Features**: TypeScript and JavaScript support, type checking, IntelliSense

## Additional Supported Languages

### C#
- **Language Server**: OmniSharp
- **Location**: `src/solidlsp/language_servers/omnisharp/`
- **Features**: Full C# language support, .NET integration

### Rust
- **Language Server**: Rust-Analyzer  
- **Location**: `src/solidlsp/language_servers/rust_analyzer/`
- **Features**: Rust language support, cargo integration

### Go
- **Language Server**: Gopls
- **Location**: `src/solidlsp/language_servers/gopls/`
- **Features**: Go language support, module system integration

### Ruby
- **Language Server**: Solargraph
- **Location**: `src/solidlsp/language_servers/solargraph/`
- **Features**: Ruby language support, gem integration

### C++
- **Language Server**: Clangd
- **Location**: `src/solidlsp/language_servers/clangd_language_server/`
- **Features**: C++ language support, CMake integration

### Dart
- **Language Server**: Dart Language Server
- **Location**: `src/solidlsp/language_servers/dart_language_server/`
- **Features**: Dart and Flutter support

### PHP
- **Language Server**: Intelephense
- **Location**: `src/solidlsp/language_servers/intelephense/`
- **Features**: PHP language support, Composer integration

### Kotlin
- **Language Server**: Kotlin Language Server
- **Location**: `src/solidlsp/language_servers/kotlin_language_server/`
- **Features**: Kotlin language support, Gradle integration

## Language Server Protocol (LSP) Integration

All language servers utilize the Language Server Protocol (LSP) to provide:
- **Semantic Analysis**: Symbol definitions, references, and relationships
- **Code Navigation**: Go-to-definition, find-references, symbol search
- **Error Detection**: Syntax and semantic error reporting
- **Code Completion**: IntelliSense-style code completion
- **Documentation**: Hover information and signature help

## Architecture Integration

### Solid-LSP Framework
- **Unified Interface**: All language servers use the same `SolidLanguageServer` interface
- **Configuration**: `LanguageServerConfig` handles language-specific settings
- **Lifecycle Management**: Start, stop, and restart capabilities for all servers
- **Process Management**: Direct process control for each language server

### Project Integration
- **Automatic Detection**: Language selection based on project file composition
- **Configuration**: Project-specific language server settings via `.serena/project.yml`
- **Ignored Paths**: Respects gitignore and custom ignore patterns
- **Timeout Settings**: Configurable per-language server timeouts

## Adding New Language Servers

The architecture supports adding new language servers by:
1. **Creating Language Server Class**: Implement language-specific server in `src/solidlsp/language_servers/`
2. **Configuration**: Add language detection and configuration rules
3. **Integration**: Register with the solid-lsp framework
4. **Testing**: Ensure semantic tools work correctly with the new language server

The system is designed to be extensible while maintaining consistent behavior across all supported languages.