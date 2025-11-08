<p align="center" style="text-align:center">
  <img src="resources/serena-logo.svg#gh-light-mode-only" style="width:500px">
  <img src="resources/serena-logo-dark-mode.svg#gh-dark-mode-only" style="width:500px">
</p>

* :rocket: Serena is a powerful **coding agent toolkit** capable of turning an LLM into a fully-featured agent that works **directly on your codebase**.
  Unlike most other tools, it is not tied to an LLM, framework or an interface, making it easy to use it in a variety of ways.
* :wrench: Serena provides essential **semantic code retrieval and editing tools** that are akin to an IDE's capabilities, extracting code entities at the symbol level and exploiting relational structure. When combined with an existing coding agent, these tools greatly enhance (token) efficiency.
* :free: Serena is **free & open-source**, enhancing the capabilities of LLMs you already have access to free of charge.

You can think of Serena as providing IDE-like tools to your LLM/coding agent. 
With it, the agent no longer needs to read entire files, perform grep-like searches or basic string replacements to find the right parts of the code and to edit code. 
Instead, it can use code-centric tools like `find_symbol`, `find_referencing_symbols` and `insert_after_symbol`.

<p align="center">
  <em>Serena is under active development! See the latest updates, upcoming features, and lessons learned to stay up to date.</em>
</p>

<p align="center">
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/Updates-1e293b?style=flat&logo=rss&logoColor=white&labelColor=1e293b" alt="Changelog" /></a>
  <a href="roadmap.md"><img src="https://img.shields.io/badge/Roadmap-14532d?style=flat&logo=target&logoColor=white&labelColor=14532d" alt="Roadmap" /></a>
  <a href="lessons_learned.md"><img src="https://img.shields.io/badge/Lessons-Learned-7c4700?style=flat&logo=readthedocs&logoColor=white&labelColor=7c4700" alt="Lessons Learned" /></a>
</p>

### LLM Integration

Serena provides the necessary [tools](#list-of-tools) for coding workflows, but an LLM is required to do the actual work,
orchestrating tool use.

For example, **supercharge the performance of Claude Code** with a [one-line shell command](#claude-code).

In general, Serena can be integrated with an LLM in several ways:

* by using the **model context protocol (MCP)**.
  Serena provides an MCP server which integrates with
    * Claude Code and Claude Desktop,
    * terminal-based clients like Codex, Gemini-CLI, Qwen3-Coder, rovodev, OpenHands CLI and others,
    * IDEs like VSCode, Cursor or IntelliJ,
    * Extensions like Cline or Roo Code
    * Local clients like [OpenWebUI](https://docs.openwebui.com/openapi-servers/mcp), [Jan](https://jan.ai/docs/mcp-examples/browser/browserbase#enable-mcp), [Agno](https://docs.agno.com/introduction/playground) and others
* by using [mcpo to connect it to ChatGPT](docs/03-special-guides/serena_on_chatgpt.md) or other clients that don't support MCP but do support tool calling via OpenAPI.
* by incorporating Serena's tools into an agent framework of your choice, as illustrated [here](docs/03-special-guides/custom_agent.md).
  Serena's tool implementation is decoupled from the framework-specific code and can thus easily be adapted to any agent framework.

### Serena in Action

#### Demonstration 1: Efficient Operation in Claude Code

A demonstration of Serena efficiently retrieving and editing code within Claude Code, thereby saving tokens and time. Efficient operations are not only useful for saving costs, but also for generally improving the generated code's quality. This effect may be less pronounced in very small projects, but often becomes of crucial importance in larger ones.

https://github.com/user-attachments/assets/ab78ebe0-f77d-43cc-879a-cc399efefd87

#### Demonstration 2: Serena in Claude Desktop

A demonstration of Serena implementing a small feature for itself (a better log GUI) with Claude Desktop.
Note how Serena's tools enable Claude to find and edit the right symbols.

https://github.com/user-attachments/assets/6eaa9aa1-610d-4723-a2d6-bf1e487ba753

### Programming Language Support & Semantic Analysis Capabilities

Serena's semantic code analysis capabilities build on **language servers** using the widely implemented
language server protocol (LSP). The LSP provides a set of versatile code querying
and editing functionalities based on symbolic understanding of the code.
Equipped with these capabilities, Serena discovers and edits code just like a seasoned developer
making use of an IDE's capabilities would.
Serena can efficiently find the right context and do the right thing even in very large and
complex projects! So not only is it free and open-source, it frequently achieves better results
than existing solutions that charge a premium.

Language servers provide support for a wide range of programming languages.
With Serena's LSP library, we provide **support for over 30 programming languages**, including
AL, Bash, C#, C/C++, Clojure, Dart, Elixir, Elm, Erlang, Fortran, Go, Haskell, Java, Javascript, Julia, Kotlin, Lua, Markdown, Nix, Perl, PHP, Python, R, Ruby, Rust, Scala, Swift, TypeScript and Zig.

> [!IMPORTANT]
> Some languages require additional dependencies to be installed; see the [Language Support](https://oraios.github.io/serena/01-about/020_programming-languages.html) page for details.

> [!TIP]
> Support for further languages can easily be added by providing a shallow adapter for a new language server implementation,
> see Serena's [memory on that](.serena/memories/adding_new_language_support_guide.md).

### Community Feedback

Most users report that Serena has strong positive effects on the results of their coding agents, even when used within
very capable agents like Claude Code. Serena is often described to be a [game changer](https://www.reddit.com/r/ClaudeAI/comments/1lfsdll/try_out_serena_mcp_thank_me_later/), providing an enormous [productivity boost](https://www.reddit.com/r/ClaudeCode/comments/1mguoia/absolutely_insane_improvement_of_claude_code).

Serena excels at navigating and manipulating complex codebases, providing tools that support precise code retrieval and editing in the presence of large, strongly structured codebases.
However, when dealing with tasks that involve only very few/small files, you may not benefit from including Serena on top of your existing coding agent. 
In particular, when writing code from scratch, Serena will not provide much value initially, as the more complex structures that Serena handles more gracefully than simplistic, file-based approaches are yet to be created.

Several videos and blog posts have talked about Serena:

* YouTube:
    * [AI Labs](https://www.youtube.com/watch?v=wYWyJNs1HVk&t=1s)
    * [Yo Van Eyck](https://www.youtube.com/watch?v=UqfxuQKuMo8&t=45s)
    * [JeredBlu](https://www.youtube.com/watch?v=fzPnM3ySmjE&t=32s)

* Blog posts:
    * [Serena's Design Principles](https://medium.com/@souradip1000/deconstructing-serenas-mcp-powered-semantic-code-understanding-architecture-75802515d116)
    * [Serena with Claude Code (in Japanese)](https://blog.lai.so/serena/)
    * [Turning Claude Code into a Development Powerhouse](https://robertmarshall.dev/blog/turning-claude-code-into-a-development-powerhouse/)

## Quick Start

Serena is managed with **uv**. If you don’t already have it installed, you will need to [install it](https://docs.astral.sh/uv/getting-started/installation/).

To get started as quickly as possible, you can directly enable the Serena MCP server in your client of choice (by following the links provided): 

* For coding with Claude, we recommend using Serena through [Claude Code](https://oraios.github.io/serena/02-usage/030_clients.html#claude-code) or [Claude Desktop](#claude-desktop). You can also use Serena in [Codex](https://oraios.github.io/serena/02-usage/030_clients.html#codex) and a wide variety of [other clients](https://oraios.github.io/serena/02-usage/030_clients.html#other-clients).
* If you want a GUI experience outside an IDE, you can use one of the many [local GUIs](https://oraios.github.io/serena/02-usage/030_clients.html#local-guis-and-agent-frameworks) that support MCP servers.
  You can also connect Serena to many web clients (including ChatGPT) using [mcpo](https://oraios.github.io/serena/03-special-guides/serena_on_chatgpt.html).
* If you want to integrate Serena with your IDE, see the section on [other MCP clients](https://oraios.github.io/serena/02-usage/030_clients.html#mcp-enabled-ides-and-coding-clients-cline-roo-code-cursor-windsurf-etc).

> [!TIP]
> While getting started quickly is easy, Serena is a powerful toolkit with many configuration options.
> We highly recommend reading through the [user guide](https://oraios.github.io/serena/) to get the most out of Serena.

## User Guide

Please refer to the [user guide](https://oraios.github.io/serena/) for detailed instructions on how to use Serena effectively.

## Acknowledgements

### Sponsors

We are very grateful to our [sponsors](https://github.com/sponsors/oraios) who help us drive Serena's development. The core team
(the founders of [Oraios AI](https://oraios-ai.de/)) put in a lot of work in order to turn Serena into a useful open source project. 
So far, there is no business model behind this project, and sponsors are our only source of income from it.

Sponsors help us dedicating more time to the project, managing contributions, and working on larger features (like better tooling based on more advanced
LSP features, VSCode integration, debugging via the DAP, and several others).
If you find this project useful to your work, or would like to accelerate the development of Serena, consider becoming a sponsor.

We are proud to announce that the Visual Studio Code team, together with Microsoft’s Open Source Programs Office and GitHub Open Source
have decided to sponsor Serena with a one-time contribution!

<p align="center">
  <img src="resources/vscode_sponsor_logo.png" alt="Visual Studio Code sponsor logo" width="220">
</p>

### Community Contributions

A significant part of Serena, especially support for various languages, was contributed by the open source community.
We are very grateful for the many contributors who made this possible and who played an important role in making Serena
what it is today.

### Technologies
We built Serena on top of multiple existing open-source technologies, the most important ones being:

1. [multilspy](https://github.com/microsoft/multilspy).
   A library which wraps language server implementations and adapts them for interaction via Python
   and which provided the basis for our library Solid-LSP (src/solidlsp).
   Solid-LSP provides pure synchronous LSP calls and extends the original library with the symbolic logic
   that Serena required.
2. [Python MCP SDK](https://github.com/modelcontextprotocol/python-sdk)
3. [Agno](https://github.com/agno-agi/agno) and
   the associated [agent-ui](https://github.com/agno-agi/agent-ui),
   which we use to allow Serena to work with any model, beyond the ones
   supporting the MCP.
4. All the language servers that we use through Solid-LSP.

Without these projects, Serena would not have been possible (or would have been significantly more difficult to build).

## Customizing and Extending Serena

It is straightforward to extend Serena's AI functionality with your own ideas.
Simply implement a new tool by subclassing
`serena.agent.Tool` and implement the `apply` method with a signature
that matches the tool's requirements.
Once implemented, `SerenaAgent` will automatically have access to the new tool.

It is also relatively straightforward to add [support for a new programming language](/.serena/memories/adding_new_language_support_guide.md).

We look forward to seeing what the community will come up with!
For details on contributing, see [contributing guidelines](/CONTRIBUTING.md).

## List of Tools

Here is the list of Serena's default tools with a short description (output of `uv run serena tools list`):

* `activate_project`: Activates a project based on the project name or path.
* `check_onboarding_performed`: Checks whether project onboarding was already performed.
* `create_text_file`: Creates/overwrites a file in the project directory.
* `delete_memory`: Deletes a memory from Serena's project-specific memory store.
* `execute_shell_command`: Executes a shell command.
* `find_file`: Finds files in the given relative paths
* `find_referencing_symbols`: Finds symbols that reference the symbol at the given location (optionally filtered by type).
* `find_symbol`: Performs a global (or local) search for symbols with/containing a given name/substring (optionally filtered by type).
* `get_current_config`: Prints the current configuration of the agent, including the active and available projects, tools, contexts, and modes.
* `get_symbols_overview`: Gets an overview of the top-level symbols defined in a given file.
* `insert_after_symbol`: Inserts content after the end of the definition of a given symbol.
* `insert_before_symbol`: Inserts content before the beginning of the definition of a given symbol.
* `list_dir`: Lists files and directories in the given directory (optionally with recursion).
* `list_memories`: Lists memories in Serena's project-specific memory store.
* `onboarding`: Performs onboarding (identifying the project structure and essential tasks, e.g. for testing or building).
* `prepare_for_new_conversation`: Provides instructions for preparing for a new conversation (in order to continue with the necessary context).
* `read_file`: Reads a file within the project directory.
* `read_memory`: Reads the memory with the given name from Serena's project-specific memory store.
* `rename_symbol`: Renames a symbol throughout the codebase using language server refactoring capabilities.
* `replace_regex`: Replaces content in a file by using regular expressions.
* `replace_symbol_body`: Replaces the full definition of a symbol.
* `search_for_pattern`: Performs a search for a pattern in the project.
* `think_about_collected_information`: Thinking tool for pondering the completeness of collected information.
* `think_about_task_adherence`: Thinking tool for determining whether the agent is still on track with the current task.
* `think_about_whether_you_are_done`: Thinking tool for determining whether the task is truly completed.
* `write_memory`: Writes a named memory (for future reference) to Serena's project-specific memory store.

There are several tools that are disabled by default, and have to be enabled explicitly, e.g., through the context or modes.
Note that several of our default contexts do enable some of these tools. For example, the `desktop-app` context enables the `execute_shell_command` tool.

The full list of optional tools is (output of `uv run serena tools list --only-optional`):

* `delete_lines`: Deletes a range of lines within a file.
* `get_current_config`: Prints the current configuration of the agent, including the active and available projects, tools, contexts, and modes.
* `initial_instructions`: Gets the initial instructions for the current project.
    Should only be used in settings where the system prompt cannot be set,
    e.g. in clients you have no control over, like Claude Desktop.
* `insert_at_line`: Inserts content at a given line in a file.
* `jet_brains_find_referencing_symbols`: Finds symbols that reference the given symbol
* `jet_brains_find_symbol`: Performs a global (or local) search for symbols with/containing a given name/substring (optionally filtered by type).
* `jet_brains_get_symbols_overview`: Retrieves an overview of the top-level symbols within a specified file
* `remove_project`: Removes a project from the Serena configuration.
* `replace_lines`: Replaces a range of lines within a file with new content.
* `restart_language_server`: Restarts the language server, may be necessary when edits not through Serena happen.
* `summarize_changes`: Provides instructions for summarizing the changes made to the codebase.
* `switch_modes`: Activates modes by providing a list of their names
