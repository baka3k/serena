<p align="center" style="text-align:center">
  <img src="resources/serena-logo.svg#gh-light-mode-only" style="width:400px">
  <img src="resources/serena-logo-dark-mode.svg#gh-dark-mode-only" style="width:400px">
</p>

> **:rocket: Serena is a powerful, fully-featured coding agent that works directly on your codebase!  
> :wrench: Serena integrates with existing LLMs and provides them with essential semantic code retrieval and editing tools!  
> :free: Serena is free to use. No additional API keys or subscriptions required!**

Q: Can I have a state-of-the-art coding agent without paying (enormous) API costs 
or constantly purchasing tokens?  
A: Yes, you can!
By integrating Serena with your favourite (even free) LLM and thereby enabling it
to perform coding tasks directly on your codebase.

Serena can be integrated with an LLM in several ways:
 * by using the **model context protocol (MCP)**.  
   Serena provides an MCP server which integrates with Claude (and [soon also ChatGPT](https://x.com/OpenAIDevs/status/1904957755829481737)).
 * by using **Agno – the model-agnostic agent framework**.
   Serena's Agno-based agent allows you to turn virtually any LLM into a coding agent, whether it's from Google, OpenAI or DeepSeek (with a paid API key)
   or completely free like Ollama, Together or Anyscale.

Serena's semantic code analysis capabilities build on **language servers** using the widely implemented
language server protocol (LSP). The LSP provides a set of versatile code querying
and editing functionalities based on symbolic understanding of the code. 
Equipped with these capabilities, Serena discovers and edits code just like a seasoned developer 
making use of an IDE's capabilities would.
Serena can efficiently find the right context and do the right thing even in very large and
complex projects! So not only is it free and open-source, it frequently achieves better results 
than existing solutions that charge a premium.

## Is It Really Free?

Yes! Even the free tier of Anthropic's Claude has support for MCP Servers, so you can use Serena there.
Presumably, the same will be true for ChatGPT Desktop once they add support for MCP servers.
But we do recommend to buy the Claude Pro subscription for 20$ per month as this way the rate
limits are much higher.

Serena is [Oraios AI](https://oraios-ai.de/)'s contribution to the developer community. We use it ourselves every day.

We got tired of having to pay multiple
subscriptions (Cursor, Windsurf) that forced us to keep purchasing tokens on top of the subscription costs.
We also got tired of paying the huge API costs from Claude Code, Cline, Aider and other API-based tools.
This is why we built serena and cancelled most subscriptions. 
See below for a more detailed comparison to existing technologies.

## What if I Do Want to Pay?

If you want to use your own API key, connect serena to a custom model,
or use it within a paid tool, you can do so.
We decoupled the tools offered by serena from the MCP server implementation, so they can also
be used in any agent framework and with any model. See the section [Serena as Agent](#serena-as-agent).
This way, you can run serena as a command line tool or within a UI of your choice.

## Quick Start

1. Install `uv` if not done yet (instructions [here](https://docs.astral.sh/uv/getting-started/installation/))
2. Clone the repository to `/path/to/serena`.
3. Create a configuration file for your project, say `myproject.yml`. See `myproject.demo.yml` for the structure. 
   You can just copy it and edit the entries.
4. Configure the mcp server in your client. For example, for Claude Desktop, you need to go to `Settings->Developer->MCP Servers`,
   which will let you open a json file. There add the following to enable serena:

```json
{
    "mcpServers": {
        "serena": {
            "command": "/abs/path/to/uv",
            "args": ["run", "--directory", "/abs/path/to/serena", "serena", "/abs/path/to/myproject.yml"]
        }
    }
}
```

That's it! Save the config, restart Claude (you will have actually terminate the process on windows, just closing the window is not enough),
and you should see the serena mcp tools in your chat interface (the small hammer icon).

Note that serena is always configured **for a single project**. To use it for another, you will have to
write a new configuration file and adjust the config in the MCP client (don't forget to restart after).

For more info on MCP servers with Claude Desktop see [here](https://modelcontextprotocol.io/quickstart/user).

## Serena as Agent

In the agent mode, serena can be used with any model, including the currently popular (and SOTA in coding)
Gemini-2.5-pro.

...

## Getting Started

You can have a local uv or docker-interpeter based setup. The repository is also 
configured to seamlessly working within a GitHub Codespace. See the instructions
for the various setup scenarios below.

Independently of how the setup was done, the virtual environment can be 
created with `uv venv` and
activated with
`source .venv/bin/activate` and the various tasks like formatting, testing, and documentation building
can be executed using `poe`. For example, `poe format` will format the code, including the 
notebooks. Just run `poe` to see the available commands.

### Python (uv) setup

You can install a virtual environment with the required as follows

1. Create a new virtual environment: `uv venv`
2. Activate the environment:
    * On Unix or MacOS: `source .venv/bin/activate`
    * On Windows: `.venv\Scripts\activate.bat`
3. Install the required packages: `uv pip install -e ".[dev]"`

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

Note: for the WSL subsystem on Windows you might need to adjust the path for the
volume.

## Model Context Protocol (MCP) Server

Serena's functionality is intended to be used via the model context protocol (MCP),
which allows for easy integration into applications like Claude Desktop and IDEs.

### Project Configuration File

The first step is to create a `.yml` configuration file for the project you want
Serena to work on:

Copy `myproject.demo.yml` to `myproject.yml` and adjust the settings to your project.

### MCP Server Configuration

Tools like Claude Desktop need to be informed about the MCP server.
They typically start the server themselves and only need to be informed how to start
the server. 
This is typically done by providing a configuration file in JSON format as follows,

```json
{
    "mcpServers": {
        "serena": {
            "command": "/path/to/uv",
            "args": ["run", "--directory", "/path/to/serena", "serena", "/path/to/myproject.yml"]
        }
    }
}
```

where you must adjust `/path/to/serena` to the actual path of the Serena repository as well as the `.yml` file
for your project configuration.
On Windows, you can specify the path in the format `C:/path/to/serena`.

**Claude Desktop**: The JSON configuration file can be found via File / Settings / Developer / Edit Config. 
    Then edit the file named `claude_desktop_config.json`.

## Contributing

Please open new issues for bugs, feature requests and extensions. See more details about the structure and
workflow in the [contributing page](docs/04_contributing/04_contributing.rst).
