## Programming Language Support & Semantic Analysis Capabilities

Serena's semantic code analysis capabilities build on **language servers** using the widely implemented
language server protocol (LSP). The LSP provides a set of versatile code querying
and editing functionalities based on symbolic understanding of the code.
Equipped with these capabilities, Serena discovers and edits code just like a seasoned developer
making use of an IDE's capabilities would.
Serena can efficiently find the right context and do the right thing even in very large and
complex projects! So not only is it free and open-source, it frequently achieves better results
than existing solutions that charge a premium.

Language servers provide support for a wide range of programming languages.
With Serena, we provide direct, out-of-the-box support for:

* Python
* TypeScript/Javascript
* PHP (uses Intelephense LSP; set `INTELEPHENSE_LICENSE_KEY` environment variable for premium features)
* Go (requires installation of gopls)
* R (requires installation of the `languageserver` R package)
* Rust (requires [rustup](https://rustup.rs/) - uses rust-analyzer from your toolchain)
* C/C++ (you may experience issues with finding references, we are working on it)
* Zig (requires installation of ZLS - Zig Language Server)
* C#
* Ruby (by default, uses [ruby-lsp](https://github.com/Shopify/ruby-lsp), specify ruby_solargraph as your language to use the previous solargraph based implementation)
* Swift
* Kotlin (uses the pre-alpha [official kotlin LS](https://github.com/Kotlin/kotlin-lsp), some issues may appear)
* Java (_Note_: startup is slow, initial startup especially so. There may be issues with java on macos and linux, we are working on it.)
* Clojure
* Dart
* Bash
* Lua (automatically downloads lua-language-server if not installed)
* Nix (requires nixd installation)
* Elixir (requires installation of NextLS and Elixir; **Windows not supported**)
* Elm (automatically downloads elm-language-server if not installed; requires Elm compiler)
* Scala (requires some [manual setup](docs/scala_setup_guide_for_serena.md); uses Metals LSP)
* Erlang (requires installation of beam and [erlang_ls](https://github.com/erlang-ls/erlang_ls), experimental, might be slow or hang)
* Perl (requires installation of Perl::LanguageServer)
* Fortran (requires installation of fortls: `pip install fortls`)
* Haskell (automatically locates HLS via ghcup, stack, or system PATH; supports Stack and Cabal projects)
* Julia
* AL
* Markdown (must be explicitly specified via `--language markdown` when generating project config, primarily useful for documentation-heavy projects)

Support for further languages can easily be added by providing a shallow adapter for a new language server implementation,
see Serena's [memory on that](.serena/memories/adding_new_language_support_guide.md).
