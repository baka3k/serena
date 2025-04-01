<p align="center" style="text-align:center">
  <img src="resources/serena-logo.svg#gh-light-mode-only" style="width:500px">
  <img src="resources/serena-logo-dark-mode.svg#gh-dark-mode-only" style="width:500px">
</p>

* :rocket: Serena is a powerful, fully-featured **coding agent that works directly on your codebase**.
* :wrench: Serena **integrates with existing LLMs**, providing them with essential **semantic code retrieval and editing tools!**
* :free: Serena is **free to use**. No additional API keys or subscriptions required!

Q: Can I have a state-of-the-art coding agent without paying (enormous) API costs 
or constantly purchasing tokens?  
A: Yes, you can!
By integrating Serena with your favourite (even free) LLM and thereby enabling it
to perform coding tasks directly on your codebase.

Serena can be integrated with an LLM in several ways:
 * by using the **model context protocol (MCP)**.  
   Serena provides an MCP server which integrates with Claude (and [soon also ChatGPT](https://x.com/OpenAIDevs/status/1904957755829481737)).
 * by using **Agno – the model-agnostic agent framework**.
   Serena's Agno-based agent allows you to turn virtually any LLM into a coding agent, whether it's provided by Google, OpenAI or DeepSeek (with a paid API key)
   or a free model provided by Ollama, Together or Anyscale.
 * by incorporating Serena's tools into an agent framework of your choice.  
   Serena's tool implementation is decoupled from the framework-specific code and can thus easily be adapted to any agent framework.

Serena's semantic code analysis capabilities build on **language servers** using the widely implemented
language server protocol (LSP). The LSP provides a set of versatile code querying
and editing functionalities based on symbolic understanding of the code. 
Equipped with these capabilities, Serena discovers and edits code just like a seasoned developer 
making use of an IDE's capabilities would.
Serena can efficiently find the right context and do the right thing even in very large and
complex projects! So not only is it free and open-source, it frequently achieves better results 
than existing solutions that charge a premium.

Language servers provide support for a wide range of programming languages.
With Serena, we provide 
 * direct, out-of-the-box support for:
     * Python 
     * Java (_Note_: startup is slow, initial startup especially so)
     * TypeScript
 * indirect support (may require some code changes/manual installation) for:
     * Ruby (untested)
     * Go (untested)
     * C# (untested)
Further languages can easily be supported by providing a shallow adapter for a new language server
implementation.

## Is It Really Free?

Yes! Even the free tier of Anthropic's Claude has support for MCP Servers, so you can use Serena with Claude for free.
Presumably, the same will soon be possible with ChatGPT Desktop once support for MCP servers is added.  
Through Agno, you furthermore have the option to use Serena with a free/open-weights model.

Serena is [Oraios AI](https://oraios-ai.de/)'s contribution to the developer community.  
We use it ourselves on a regular basis.

We got tired of having to pay multiple
IDE-based subscriptions (such as Windsurf or Cursor) that forced us to keep purchasing tokens on top of the chat subscription costs we already had.
The substantial API costs incurred by tools like Claude Code, Cline, Aider and other API-based tools are similarly unattractive.
We thus built Serena with the prospect of being able to cancel most other subscriptions.

## Quick Start

### MCP Server (Claude Desktop)

1. Install `uv` (instructions [here](https://docs.astral.sh/uv/getting-started/installation/))
2. Clone the repository to `/path/to/serena`.
3. Create a configuration file for your project, say `myproject.yml` based on the template in [myproject.demo.yml](myproject.demo.yml).
4. Configure the MCP server in your client.  
   For Claude Desktop, go to File / Settings / Developer / MCP Servers / Edit Config,
   which will let you open the json file `claude_desktop_config.json`. Add the following (with adjusted paths) to enable Serena:

```json
{
    "mcpServers": {
        "serena": {
            "command": "/abs/path/to/uv",
            "args": ["run", "--directory", "/abs/path/to/serena", "serena-mcp-server", "/abs/path/to/myproject.yml"]
        }
    }
}
```

When using paths containing backslashes on Windows, be sure to escape them correctly (`\\`).

That's it! Save the config and then restart Claude Desktop (be sure to fully quit the application, as closing Claude will just minimize it to the system tray).
You should then see the Serena MCP tools in your chat interface (notice the small hammer icon).

Note that Serena is always configured *for a single project*. To use it for another, you will have to
write a new configuration file and adjust the config in the MCP client.

For more information on MCP servers with Claude Desktop, see [the official quick start guide](https://modelcontextprotocol.io/quickstart/user).

### Agno

Agno is a model-agnostic agent framework that allows you to use Serena with a large number of underlying LLMs.

While Agno is not yet entirely stable, we chose it, because it comes with its own open-source UI, 
making it easy to directly use the agent.  

Here's how it works (see also [Agno's documentation](https://docs.agno.com/introduction/playground)):

1. Download the agent-ui code with npx
   ```shell
   npx create-agent-ui@latest
   ```
   or, alternatively, clone it manually:
   ```shell
   git clone https://github.com/agno-agi/agent-ui.git
   cd agent-ui 
   pnpm install 
   pnpm dev
   ```

2. Install serena with the optional requirements:
   ```shell
   # You can also only select agno,google or agno,anthropic instead of all-extras
   uv pip install --all-extras -e .
   ```
   
3. Copy `.env.example` to `.env` and fill in the API keys for the provider(s) you
   intend to use.

5. Start the agno agent app with
   ```shell
   uv run python scripts/agno_agent.py
   ```
   By default, the script uses Claude as the model, but you can choose any model
   supported by Agno (which is essentially any existing model).

5. In a new terminal, start the agno UI with
   ```shell
   cd agent-ui 
   pnpm dev
   ```
   Connect the UI to the agent you started above and start chatting. You will have
   the same tools as in the MCP server version.

IMPORTANT: Contrary to the MCP server approach, tool execution in the Agno UI does
not ask for the user's permission. Note that the shell
tool can perform arbitrary code execution. While we have never encountered any issues with
this in our testing with Claude, this may not be entirely safe. 
You may choose to disable certain tools for your setup in your Serena project's
configuration file (`myproject.yml`).

## Serena's Tools and Configuration

Serena combines tools for semantic code retrieval with editing capabilities and shell execution.

The use of all tools is generally recommended, as this allows Serena to provide the most value:
Only by executing shell commands (in particular, tests) can Serena identify and correct mistakes 
autonomously.

However, it should be noted that the `execute_shell_command` tool allows for arbitrary code execution. 
When using Serena as an MCP Server, clients will typically ask the user for permission 
before executing a tool, so as long as the user inspects execution parameters beforehand,
this should not be a problem.
However, if you have concerns, you can choose to disable certain commands in your project's 
.yml configuration file.
If you only want to use Serena purely for analyzing code and suggesting implementations
without modifying the codebase, you can consider disabling the editing tools in the configuration, i.e.

  * `create_text_file`
  * `insert_after_symbol`
  * `insert_at_line`
  * `insert_before_symbol`
  * `replace_symbol_body`
  * `delete_lines`.

In general, be sure to back up your work and use a version control system in order to avoid
losing any work.

Here a complete list of Serena's tools:

* `check_onboarding_performed`: Checks whether the onboarding was already performed.
* `create_text_file`: Creates/overwrites a file in the project directory.
* `delete_lines`: Deletes a range of lines within a file.
* `delete_memory`: Deletes a memory from Serena's project-specific memory store.
* `execute_shell_command`: Executes a shell command.
* `find_referencing_symbols`: Finds symbols that reference the symbol at the given location (optionally filtered by type).
* `find_symbol`: Performs a global (or local) search for symbols with/containing a given name/substring (optionally filtered by type).
* `get_dir_overview`: Gets an overview of the top-level symbols defined in all files within a given directory.
* `get_document_overview`: Gets an overview of the top-level symbols defined in a given file.
* `insert_after_symbol`: Inserts content after the end of the definition of a given symbol.
* `insert_at_line`: Inserts content at a given line in a file.
* `insert_before_symbol`: Inserts content before the beginning of the definition of a given symbol.
* `list_dir`: Lists files and directories in the given directory (optionally with recursion).
* `list_memories`: Lists memories in Serena's project-specific memory store.
* `onboarding`: Performs onboarding (identifying the project structure and essential tasks, e.g. for testing or building).
* `prepare_for_new_conversation`: Provides instructions for preparing for a new conversation (in order to continue with the necessary context).
* `read_file`: Reads a file within the project directory.
* `read_memory`: Reads the memory with the given name from Serena's project-specific memory store.
* `replace_symbol_body`: Replaces the full definition of a symbol.
* `search_in_all_code`: Performs a search for a pattern in all code files (and only in code files) in the project.
* `summarize_changes`: Provides instructions for summarizing the changes made to the codebase.
* `think_about_collected_information`: Thinking tool for pondering the completeness of collected information.
* `think_about_task_adherence`: Thinking tool for determining whether the agent is still on track with the current task.
* `think_about_whether_you_are_done`: Thinking tool for determining whether the task is truly completed.
* `write_memory`: Writes a named memory (for future reference) to Serena's project-specific memory store.

## Comparison with Other Coding Agents

To our knowledge, Serena is the first fully-featured coding agent where the
entire functionality
is available through an MCP server, thus not requiring API keys or
subscriptions.
Here a brief comparison with other tools:

### Subscription-Based Coding Agents

The most prominent subscription-based coding agents are parts of IDEs like
Windsurf, Cursor and VSCode.
Serena's functionality is similar to Cursor's Agent, Windsurf's Cascade or
VSCode's
upcoming [agent mode](https://code.visualstudio.com/blogs/2025/02/24/introducing-copilot-agent-mode).

Serena has the advantage of not requiring a subscription.
A potential disadvantage is that it 
is not directly integrated into an IDE, so the inspection of newly written code
is not as seamless.

More technical differences are:
* Serena is not bound to a specific IDE.
  Serena's MCP server can be used with any MCP client (including some IDEs), 
  and the Agno-based agent provides additional ways of applying its functionality.
* Serena is not bound to a specific large language model or API.
* Serena navigates and edits code using a language server, so it has a symbolic
  understanding of the code.
  IDE-based tools often use a RAG-based or purely text-based approach, which is often
  less powerful, especially for large codebases.
* Serena is open-source and has a small codebase, so it can be easily extended
  and modified.

### API-Based Coding Agents

An alternative to subscription-based agents are API-based agents like Claude
Code, Cline, Aider, Roo Code and others, where the usage costs map directly
to the API costs of the underlying LLM.
Some of them (like Cline) can even be included in IDEs as an extension.
They are often very powerful and their main downside are the (potentially very
high) API costs.

Serena itself can be used as an API-based agent (see the section on Agno above).
We have not yet written a CLI tool or a
dedicated IDE extension for Serena (and there is probably no need for the latter, as
Serena can already be used with any IDE that supports MCP servers).
If there is demand for a Serena as a CLI tool like Claude Code, we will
consider writing one.

The main difference between Serena and other API-based agents is that Serena can
also be used as an MCP server, thus not requiring
an API key and bypassing the API costs. This is a unique feature of Serena.

### Other MCP-Based Coding Agents

There are other MCP servers meant for coding, like for
example [DesktopCommander](https://github.com/wonderwhy-er/DesktopCommanderMCP) and
[codemcp](https://github.com/ezyang/codemcp).
However, to the best of our knowledge, none of them provide semantic code
retrieval and editing tools; they rely purely on text-based analysis. 
It is the integration of language servers and the MCP that makes Serena unique 
and so powerful for challenging coding tasks, especially in the context of
larger codebases.

## Onboarding and Memories

By default, Serena will perform an onboarding process when 
it is started for the first time for a project. It will save some information
as memories to `.serena/memories/` in the project directory.

The memories are human-readable and will be used as references for further tasks.
Feel free to read and adjust them as needed, you can also add new ones manually.
Every file in the `.serena/memories/` directory is a memory file.

We found the memories to significantly improve the user experience with Serena.
By itself, Serena is instructed to create new memories whenever appropriate.

## Recommendations on Using Serena

### Before Editing Code

### Running Out of Context

### Controlling Tool Execution

### Structuring the Codebase

## Developer Environment Setup

You can have a local setup via `uv` or a docker interpreter-based setup. 
The repository is also configured to seamlessly work within a GitHub Codespace. See the instructions
for the various setup scenarios below.

Independently of how the setup was done, the virtual environment can be 
created and activated via `uv` (see below), and the various tasks like formatting, testing, and documentation building
can be executed using `poe`. For example, `poe format` will format the code, including the 
notebooks. Just run `poe` to see the available commands.

### Python (uv) setup

You can install a virtual environment with the required as follows

1. Create a new virtual environment: `uv venv`
2. Activate the environment:
    * On Linux/Unix/macOS: `source .venv/bin/activate`
    * On Windows: `.venv\Scripts\activate.bat`
3. Install the required packages: `uv pip install --all-extras -r pyproject.toml -e .`

### Docker setup

Build the docker image with

```shell
docker build -t serena .
```

and run it with the repository mounted as a volume:

```shell
docker run -it --rm -v "$(pwd)":/workspace serena
```

You can also just run `bash docker_build_and_run.sh`, which will do both things
for you.

Note: For the Windows subsystem for Linux (WSL), you may need to adjust the path for the
volume.

## Acknowledgements

We built Serena on top of multiple existing open-source technologies, to which we are very grateful.
Here a short list of the most important ones:

1. [Multilspy](https://github.com/microsoft/multilspy).
   A beautifully designed wrapper around language servers following the LSP. It
   was not easily extendable with the symbolic
   logic that Serena required, so instead of incorporating it as dependency, we
   copied the source code
   and adapted it to our needs.
2. [Python MCP SDK](https://github.com/modelcontextprotocol/python-sdk)
3. [Agno](https://github.com/agno-agi/agno) and
   agno's [agent-ui](https://github.com/agno-agi/agent-ui),
   which we use to allow Serena to work with any model, not just those that
   support the MCP.
4. All the language servers that we use through multilspy.

Without these projects, Serena would not have been possible, or would at least much more difficult to build.


## Contributing

Please open new issues for bugs, feature requests and extensions. See more details about the structure and
workflow in the [contributing page](docs/04_contributing/04_contributing.rst).
