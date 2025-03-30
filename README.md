<p align="center" style="text-align:center">
  <img src="resources/serena-logo.svg#gh-light-mode-only" style="width:500px">
  <img src="resources/serena-logo-dark-mode.svg#gh-dark-mode-only" style="width:500px">
</p>

> * :rocket: **Serena is a powerful, fully-featured coding agent that works directly on your codebase.**  
> * :wrench: **Serena integrates with existing LLMs and provides them with essential semantic code retrieval and editing tools!**
> * :free: **Serena is free to use. No additional API keys or subscriptions required!**

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
   or completely, e.g. Ollama, Together or Anyscale.

Serena's semantic code analysis capabilities build on **language servers** using the widely implemented
language server protocol (LSP). The LSP provides a set of versatile code querying
and editing functionalities based on symbolic understanding of the code. 
Equipped with these capabilities, Serena discovers and edits code just like a seasoned developer 
making use of an IDE's capabilities would.
Serena can efficiently find the right context and do the right thing even in very large and
complex projects! So not only is it free and open-source, it frequently achieves better results 
than existing solutions that charge a premium.

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
            "args": ["run", "--directory", "/abs/path/to/serena", "serena", "/abs/path/to/myproject.yml"]
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

With Agno, Serena can be used with any model, including the currently popular (and SOTA in coding)
Gemini-2.5-pro.

...

## Developer Environment Setup

You can have a local setup via `uv` or a docker interpeter-based setup. 
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

Note: For the Windows subsystem for Linux (WSL), you may need to adjust the path for the
volume.

## Contributing

Please open new issues for bugs, feature requests and extensions. See more details about the structure and
workflow in the [contributing page](docs/04_contributing/04_contributing.rst).
