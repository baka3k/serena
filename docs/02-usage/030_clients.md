## Configuring Your MCP Client

### Claude Code

Serena is a great way to make Claude Code both cheaper and more powerful!

From your project directory, add serena with a command like this,

```shell
claude mcp add serena -- <serena-mcp-server> --context ide-assistant --project "$(pwd)"
```

where `<serena-mcp-server>` is your way of [running the Serena MCP server](#running-the-serena-mcp-server).
For example, when using `uvx`, you would run

```shell
claude mcp add serena -- uvx --from git+https://github.com/oraios/serena serena start-mcp-server --context ide-assistant --project "$(pwd)"
```

ℹ️ Serena comes with an instruction text, and Claude needs to read it to properly use Serena's tools.
As of version `v1.0.52`, claude code reads the instructions of the MCP server, so this **is handled automatically**.
If you are using an older version, or if Claude fails to read the instructions, you can ask it explicitly
to "read Serena's initial instructions" or run `/mcp__serena__initial_instructions` to load the instruction text.
If you want to make use of that, you will have to enable the corresponding tool explicitly by adding `initial_instructions` to the `included_optional_tools`
in your config.
Note that you may have to make Claude read the instructions when you start a new conversation and after any compacting operation to ensure Claude remains properly configured to use Serena's tools.

### Codex

Serena works with OpenAI's Codex CLI out of the box, but you have to use the `codex` context for it to work properly. (The technical reason is that Codex doesn't fully support the MCP specifications, so some massaging of tools is required.).

Unlike Claude Code, in Codex you add an MCP server globally and not per project. Add the following to
`~/.codex/config.toml` (create the file if it does not exist):

```toml
[mcp_servers.serena]
command = "uvx"
args = ["--from", "git+https://github.com/oraios/serena", "serena", "start-mcp-server", "--context", "codex"]
```

After codex has started, you need to activate the project, which you can do by saying:

"Activate the current dir as project using serena"

> If you don't activate the project, you will not be able to use Serena's tools!

That's it! Have a look at `~/.codex/log/codex-tui.log` to see if any errors occurred.

The Serena dashboard will run if you have not disabled it in the configuration, but due to Codex's sandboxing the webbrowser
may not open automatically. You can open it manually by going to `http://localhost:24282/dashboard/index.html` (or a higher port, if
that was already taken).

> Codex will often show the tools as `failed` even though they are successfully executed. This is not a problem, seems to be a bug in Codex. Despite the error message, everything works as expected.

### Other Terminal-Based Clients

There are many terminal-based coding assistants that support MCP servers, such as [Codex](https://github.com/openai/codex?tab=readme-ov-file#model-context-protocol-mcp),
[Gemini-CLI](https://github.com/google-gemini/gemini-cli), [Qwen3-Coder](https://github.com/QwenLM/Qwen3-Coder),
[rovodev](https://community.atlassian.com/forums/Rovo-for-Software-Teams-Beta/Introducing-Rovo-Dev-CLI-AI-Powered-Development-in-your-terminal/ba-p/3043623),
the [OpenHands CLI](https://docs.all-hands.dev/usage/how-to/cli-mode) and [opencode](https://github.com/sst/opencode).

They generally benefit from the symbolic tools provided by Serena. You might want to customize some aspects of Serena
by writing your own context, modes or prompts to adjust it to your workflow, to other MCP servers you are using, and to
the client's internal capabilities.

### Claude Desktop

For [Claude Desktop](https://claude.ai/download) (available for Windows and macOS), go to File / Settings / Developer / MCP Servers / Edit Config,
which will let you open the JSON file `claude_desktop_config.json`.
Add the `serena` MCP server configuration, using a [run command](#running-the-serena-mcp-server) depending on your setup.

* local installation:

   ```json
   {
       "mcpServers": {
           "serena": {
               "command": "/abs/path/to/uv",
               "args": ["run", "--directory", "/abs/path/to/serena", "serena", "start-mcp-server"]
           }
       }
   }
   ```

* uvx:

   ```json
   {
       "mcpServers": {
           "serena": {
               "command": "/abs/path/to/uvx",
               "args": ["--from", "git+https://github.com/oraios/serena", "serena", "start-mcp-server"]
           }
       }
  }
  ```

* docker:

  ```json
   {
       "mcpServers": {
           "serena": {
               "command": "docker",
               "args": ["run", "--rm", "-i", "--network", "host", "-v", "/path/to/your/projects:/workspaces/projects", "ghcr.io/oraios/serena:latest", "serena", "start-mcp-server", "--transport", "stdio"]
           }
       }
   }
   ```

If you are using paths containing backslashes for paths on Windows
(note that you can also just use forward slashes), be sure to escape them correctly (`\\`).

That's it! Save the config and then restart Claude Desktop. You are ready for activating your first project.

ℹ️ You can further customize the run command using additional arguments (see [above](#command-line-arguments)).

Note: on Windows and macOS there are official Claude Desktop applications by Anthropic, for Linux there is an [open-source
community version](https://github.com/aaddrick/claude-desktop-debian).

⚠️ Be sure to fully quit the Claude Desktop application, as closing Claude will just minimize it to the system tray – at least on Windows.

⚠️ Some clients may leave behind zombie processes. You will have to find and terminate them manually then.
With Serena, you can activate the [dashboard](#serenas-logs-the-dashboard-and-gui-tool) to prevent unnoted processes and also use the dashboard
for shutting down Serena.

After restarting, you should see Serena's tools in your chat interface (notice the small hammer icon).

For more information on MCP servers with Claude Desktop, see [the official quick start guide](https://modelcontextprotocol.io/quickstart/user).

### MCP Coding Clients (Cline, Roo-Code, Cursor, Windsurf, etc.)

Being an MCP Server, Serena can be included in any MCP Client. The same configuration as above,
perhaps with small client-specific modifications, should work. Most of the popular
existing coding assistants (IDE extensions or VSCode-like IDEs) support connections
to MCP Servers. It is **recommended to use the `ide-assistant` context** for these integrations by adding `"--context", "ide-assistant"` to the `args` in your MCP client's configuration. Including Serena generally boosts their performance
by providing them tools for symbolic operations.

In this case, the billing for the usage continues to be controlled by the client of your choice
(unlike with the Claude Desktop client). But you may still want to use Serena through such an approach,
e.g., for one of the following reasons:

1. You are already using a coding assistant (say Cline or Cursor) and just want to make it more powerful.
2. You are on Linux and don't want to use the [community-created Claude Desktop](https://github.com/aaddrick/claude-desktop-debian).
3. You want tighter integration of Serena into your IDE and don't mind paying for that.

### Local GUIs and Frameworks

Over the last months, several technologies have emerged that allow you to run a powerful local GUI
and connect it to an MCP server. They will work with Serena out of the box.
Some of the leading open source GUI technologies offering this are
[Jan](https://jan.ai/docs/mcp), [OpenHands](https://github.com/All-Hands-AI/OpenHands/),
[OpenWebUI](https://docs.openwebui.com/openapi-servers/mcp) and [Agno](https://docs.agno.com/introduction/playground).
They allow combining Serena with almost any LLM (including locally running ones) and offer various other integrations.
