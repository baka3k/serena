## Running Serena

Serena is a command-line tool with a variety of sub-commands.
This section describes
 * varies ways of running Serena
 * how to run and configure the most important command, i.e. starting the MCP server
 * other useful commands.

### Ways of Running Serena

In the following, we will refer to the command used to run Serena as `<serena>`,
which you should replace with the appropriate command based on your chosen method.

#### Using uvx

`uvx` is part of `uv`. It can be used to run the latest version of Serena directly from the repository, without an explicit local installation.

    uvx --from git+https://github.com/oraios/serena serena 

Explore the CLI to see some of the customization options that serena provides (more info on them below).

#### Local Installation

1. Clone the repository and change into it.

   ```shell
   git clone https://github.com/oraios/serena
   cd serena
   ```

2. Run Serena via

   ```shell
   uv run serena 
   ```

   when within the serena installation directory.   
   From other directories, run it with the `--directory` option, i.e.

   ```shell
    uv run --directory /abs/path/to/serena serena
    ```

(docker)=
#### Using Docker (Experimental)

⚠️ Docker support is currently experimental with several limitations. Please read the [Docker documentation](https://github.com/oraios/serena/blob/main/DOCKER.md) for important caveats before using it.

You can run the Serena MCP server directly via docker as follows,
assuming that the projects you want to work on are all located in `/path/to/your/projects`:

```shell
docker run --rm -i --network host -v /path/to/your/projects:/workspaces/projects ghcr.io/oraios/serena:latest serena 
```

Replace `/path/to/your/projects` with the absolute path to your projects directory. The Docker approach provides:

* Better security isolation for shell command execution
* No need to install language servers and dependencies locally
* Consistent environment across different systems

Alternatively, use docker compose with the `compose.yml` file provided in the repository.

See the [Docker documentation](https://github.com/oraios/serena/blob/main/DOCKER.md) for detailed setup instructions, configuration options, and known limitations.

#### Using Nix

If you are using Nix and [have enabled the `nix-command` and `flakes` features](https://nixos.wiki/wiki/flakes), you can run Serena using the following command:

```bash
nix run github:oraios/serena -- <command> [options]
```

You can also install Serena by referencing this repo (`github:oraios/serena`) and using it in your Nix flake. The package is exported as `serena`.

### Running the MCP Server

Given your preferred method of running Serena, you can start the MCP server using the `start-mcp-server` command:

    <serena> start-mcp-server [options]  

Note that no matter how you run the MCP server, Serena will, by default, start a web-based dashboard on localhost that will allow you to inspect
the server's operations, logs, and configuration.

#### Standard I/O Mode

The typical usage involves the client (e.g. Claude Code, Codex or Cursor) running
the MCP server as a subprocess and using the process' stdin/stdout streams to communicate with it.
In order to launch the server, the client thus needs to be provided with the command to run the MCP server.

Communication over stdio is the default for the Serena MCP server, so in the simplest
case, you can simply run the `start-mcp-server` command without any additional options.
 
    <serena> start-mcp-server

For example, to run the server in stdio mode via `uvx`, you would run:

    uvx --from git+https://github.com/oraios/serena serena start-mcp-server 

ℹ️ See the section ["Configuring Your MCP Client"](030_clients) for information on how to configure your MCP client (e.g. Claude Code, Codex, Cursor, etc.)
to connect to the Serena MCP server.

ℹ️ Note that MCP servers which use stdio as a protocol are somewhat unusual as far as client/server architectures go, as the server
necessarily has to be started by the client in order for communication to take place via the server's standard input/output stream.
In other words, you do not need to start the server yourself. The client application (e.g. Claude Desktop) takes care of this and
therefore needs to be configured with a launch command.

#### Streamable HTTP Mode

When using instead the *Streamable HTTP* mode, you control the server lifecycle yourself,
i.e. you start the server and provide the client with the URL to connect to it.

Simply provide `start-mcp-server` with the `--transport streamable-http` option and optionally provide the desired port
via the `--port` option.

    <serena> start-mcp-server --transport streamable-http --port <port>

For example, to run the Serena MCP server in streamable HTTP mode on port 9121 using uvx,
you would run

    uvx --from git+https://github.com/oraios/serena serena start-mcp-server --transport streamable-http --port 9121

and then configure your client to connect to `http://localhost:9121/mcp`.

ℹ️ Note that while SSE transport is also supported, its use is discouraged.

(mcp-args)=
#### MCP Server Command-Line Arguments

The Serena MCP server supports a wide range of additional command-line options.
Use the command

    <serena> start-mcp-server --help

to get a list of all available options.

Some useful options include:

  * `--project <path|name>`: specify the project to work on by name or path.
  * `--context <context>`: specify the operation [context](contexts) in which Serena shall operate
  * `--mode <mode>`: specify one or more [modes](modes) to enable (can be passed several times)
  * `--enable-web-dashboard <true|false>`: enable or disable the web dashboard (enabled by default)

### Other Commands

Serena provides several other commands in addition to `start-mcp-server`, 
most of which are related to project setup and configuration.

To get a list of available commands, run:

    <serena> --help

To get help on a specific command, run:

    <serena> <command> --help

Here are some examples of commands you might find useful:

    # TODO
