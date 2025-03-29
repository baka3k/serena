# serena

Welcome to the serena library!

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
