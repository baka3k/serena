## Quick Start

Serena can be used in various ways, below you will find instructions for selected integrations.

* For coding with Claude, we recommend using Serena through [Claude Code](#claude-code) or [Claude Desktop](#claude-desktop). You can also use Serena in most other [terminal-based clients](#other-terminal-based-clients).
* If you want a GUI experience outside an IDE, you can use one of the many [local GUIs](#local-guis-and-frameworks) that support MCP servers.
  You can also connect Serena to many web clients (including ChatGPT) using [mcpo](docs/serena_on_chatgpt.md).
* If you want to use Serena integrated in your IDE, see the section on [other MCP clients](#other-mcp-clients---cline-roo-code-cursor-windsurf-etc).
* You can use Serena as a library for building your own applications. We try to keep the public API stable, but you should still
  expect breaking changes and pin Serena to a fixed version if you use it as a dependency.

Serena is managed by `uv`, so you will need to [install it](https://docs.astral.sh/uv/getting-started/installation/).

### Running the Serena MCP Server

You have several options for running the MCP server, which are explained in the subsections below.

#### Usage

The typical usage involves the client (Claude Code, Claude Desktop, etc.) running
the MCP server as a subprocess (using stdio communication),
so the client needs to be provided with the command to run the MCP server.
(Alternatively, you can run the MCP server in Streamable HTTP or SSE mode and tell your client
how to connect to it.)

Note that no matter how you run the MCP server, Serena will, by default, start a small web-based dashboard on localhost that will display logs and allow shutting down the
MCP server (since many clients fail to clean up processes correctly).
This and other settings can be adjusted in the [configuration](#configuration) and/or by providing [command-line arguments](#command-line-arguments).

#### Using uvx

`uvx` can be used to run the latest version of Serena directly from the repository, without an explicit local installation.

```shell
uvx --from git+https://github.com/oraios/serena serena start-mcp-server
```

Explore the CLI to see some of the customization options that serena provides (more info on them below).

#### Local Installation

1. Clone the repository and change into it.

   ```shell
   git clone https://github.com/oraios/serena
   cd serena
   ```

2. Optionally edit the configuration file in your home directory with

   ```shell
   uv run serena config edit
   ```

   If you just want the default config, you can skip this part, and a config file will be created when you first run Serena.
3. Run the server with `uv`:

   ```shell
   uv run serena start-mcp-server
   ```

   When running from outside the serena installation directory, be sure to pass it, i.e., use

   ```shell
    uv run --directory /abs/path/to/serena serena start-mcp-server
   ```

#### Using Docker (Experimental)

⚠️ Docker support is currently experimental with several limitations. Please read the [Docker documentation](DOCKER.md) for important caveats before using it.

You can run the Serena MCP server directly via docker as follows,
assuming that the projects you want to work on are all located in `/path/to/your/projects`:

```shell
docker run --rm -i --network host -v /path/to/your/projects:/workspaces/projects ghcr.io/oraios/serena:latest serena start-mcp-server --transport stdio
```

Replace `/path/to/your/projects` with the absolute path to your projects directory. The Docker approach provides:

* Better security isolation for shell command execution
* No need to install language servers and dependencies locally
* Consistent environment across different systems

Alternatively, use docker compose with the `compose.yml` file provided in the repository.

See the [Docker documentation](DOCKER.md) for detailed setup instructions, configuration options, and known limitations.

#### Using Nix

If you are using Nix and [have enabled the `nix-command` and `flakes` features](https://nixos.wiki/wiki/flakes), you can run Serena using the following command:

```bash
nix run github:oraios/serena -- start-mcp-server --transport stdio
```

You can also install Serena by referencing this repo (`github:oraios/serena`) and using it in your Nix flake. The package is exported as `serena`.

#### Streamable HTTP Mode

ℹ️ Note that MCP servers which use stdio as a protocol are somewhat unusual as far as client/server architectures go, as the server
necessarily has to be started by the client in order for communication to take place via the server's standard input/output stream.
In other words, you do not need to start the server yourself. The client application (e.g. Claude Desktop) takes care of this and
therefore needs to be configured with a launch command.

When using instead the *Streamable HTTP* mode, you control the server lifecycle yourself,
i.e. you start the server and provide the client with the URL to connect to it.

Simply provide `start-mcp-server` with the `--transport streamable-http` option and optionally provide the port.
For example, to run the Serena MCP server in Streamable HTTP mode on port 9121 using a local installation,
you would run this command from the Serena directory,

```shell
uv run serena start-mcp-server --transport streamable-http --port 9121
```

and then configure your client to connect to `http://localhost:9121/mcp`.

ℹ️ Note that SSE transport is supported as well, but its use is discouraged.
Use Streamable HTTP instead.

#### Command-Line Arguments

The Serena MCP server supports a wide range of additional command-line options, including the option to run in Streamable HTTP or SSE mode
and to adapt Serena to various [contexts and modes of operation](#modes-and-contexts).

Run with parameter `--help` to get a list of available options.

### Configuration

Serena is very flexible in terms of configuration. While for most users, the default configurations will work,
you can fully adjust it to your needs by editing a few yaml files. You can disable tools, change Serena's instructions
(what we denote as the `system_prompt`), adjust the output of tools that just provide a prompt, and even adjust tool descriptions.

Serena is configured in four places:

1. The `serena_config.yml` for general settings that apply to all clients and projects.
   It is located in your user directory under `.serena/serena_config.yml`.
   If you do not explicitly create the file, it will be auto-generated when you first run Serena.
   You can edit it directly or use

   ```shell
   uvx --from git+https://github.com/oraios/serena serena config edit
   ```

   (or use the `--directory` command version).
2. In the arguments passed to the `start-mcp-server` in your client's config (see below),
   which will apply to all sessions started by the respective client. In particular, the [context](#contexts) parameter
   should be set appropriately for Serena to be best adjusted to existing tools and capabilities of your client.
   See for a detailed explanation. You can override all entries from the `serena_config.yml` through command line arguments.
3. In the `.serena/project.yml` file within your project. This will hold project-level configuration that is used whenever
   that project is activated. This file will be autogenerated when you first use Serena on that project, but you can also
   generate it explicitly with

   ```shell
   uvx --from git+https://github.com/oraios/serena serena project generate-yml
   ```

   (or use the `--directory` command version).
4. Through the context and modes. Explore the [modes and contexts](#modes-and-contexts) section for more details.

After the initial setup, continue with one of the sections below, depending on how you
want to use Serena.

### Project Activation & Indexing

If you are mostly working with the same project, you can configure to always activate it at startup
by passing `--project <path_or_name>` to the `start-mcp-server` command in your client's MCP config.
This is especially useful for clients which configure MCP servers on a per-project basis, like Claude Code.

Otherwise, the recommended way is to just ask the LLM to activate a project by providing it an absolute path to, or,
in case the project was activated in the past, by its name. The default project name is the directory name.

* "Activate the project /path/to/my_project"
* "Activate the project my_project"

All projects that have been activated will be automatically added to your `serena_config.yml`, and for each
project, the file `.serena/project.yml` will be generated. You can adjust the latter, e.g., by changing the name
(which you refer to during the activation) or other options. Make sure to not have two different projects with the
same name.

ℹ️ For larger projects, we recommend that you index your project to accelerate Serena's tools; otherwise the first
tool application may be very slow.
To do so, run this from the project directory (or pass the path to the project as an argument):

```shell
uvx --from git+https://github.com/oraios/serena serena project index
```

(or use the `--directory` command version).

### Configuring Popular Clients

#### Claude Code

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

#### Codex

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

#### Other Terminal-Based Clients

There are many terminal-based coding assistants that support MCP servers, such as [Codex](https://github.com/openai/codex?tab=readme-ov-file#model-context-protocol-mcp),
[Gemini-CLI](https://github.com/google-gemini/gemini-cli), [Qwen3-Coder](https://github.com/QwenLM/Qwen3-Coder),
[rovodev](https://community.atlassian.com/forums/Rovo-for-Software-Teams-Beta/Introducing-Rovo-Dev-CLI-AI-Powered-Development-in-your-terminal/ba-p/3043623),
the [OpenHands CLI](https://docs.all-hands.dev/usage/how-to/cli-mode) and [opencode](https://github.com/sst/opencode).

They generally benefit from the symbolic tools provided by Serena. You might want to customize some aspects of Serena
by writing your own context, modes or prompts to adjust it to your workflow, to other MCP servers you are using, and to
the client's internal capabilities.

#### Claude Desktop

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

#### MCP Coding Clients (Cline, Roo-Code, Cursor, Windsurf, etc.)

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

#### Local GUIs and Frameworks

Over the last months, several technologies have emerged that allow you to run a powerful local GUI
and connect it to an MCP server. They will work with Serena out of the box.
Some of the leading open source GUI technologies offering this are
[Jan](https://jan.ai/docs/mcp), [OpenHands](https://github.com/All-Hands-AI/OpenHands/),
[OpenWebUI](https://docs.openwebui.com/openapi-servers/mcp) and [Agno](https://docs.agno.com/introduction/playground).
They allow combining Serena with almost any LLM (including locally running ones) and offer various other integrations.
