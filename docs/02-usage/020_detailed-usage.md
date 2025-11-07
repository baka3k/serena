## Detailed Usage and Recommendations

### Tool Execution

Serena combines tools for semantic code retrieval with editing capabilities and shell execution.
Serena's behavior can be further customized through [Modes and Contexts](#modes-and-contexts).
Find the complete list of tools [below](#full-list-of-tools).

The use of all tools is generally recommended, as this allows Serena to provide the most value:
Only by executing shell commands (in particular, tests) can Serena identify and correct mistakes
autonomously.

#### Shell Execution and Editing Tools

Many clients have their own shell execution tool, and by default Serena's shell tool will be disabled in them
(e.g., when using the `ide-assistant` or `codex` context). However, when using Serena through something like
Claude Desktop or ChatGPT, it is recommended to enable Serena's `execute_shell_command` tool to allow
agentic behavior.

It should be noted that the `execute_shell_command` tool allows for arbitrary code execution.
When using Serena as an MCP Server, clients will typically ask the user for permission
before executing a tool, so as long as the user inspects execution parameters beforehand,
this should not be a problem.
However, if you have concerns, you can choose to disable certain commands in your project's configuration file.
If you only want to use Serena purely for analyzing code and suggesting implementations
without modifying the codebase, you can enable read-only mode by setting `read_only: true` in your project configuration file.
This will automatically disable all editing tools and prevent any modifications to your codebase while still
allowing all analysis and exploration capabilities.

In general, be sure to back up your work and use a version control system in order to avoid
losing any work.

### Modes and Contexts

Serena's behavior and toolset can be adjusted using contexts and modes.
These allow for a high degree of customization to best suit your workflow and the environment Serena is operating in.

#### Contexts

A context defines the general environment in which Serena is operating.
It influences the initial system prompt and the set of available tools.
A context is set at startup when launching Serena (e.g., via CLI options for an MCP server or in the agent script) and cannot be changed during an active session.

Serena comes with pre-defined contexts:

* `desktop-app`: Tailored for use with desktop applications like Claude Desktop. This is the default.
* `agent`: Designed for scenarios where Serena acts as a more autonomous agent, for example, when used with Agno.
* `ide-assistant`: Optimized for integration into IDEs like VSCode, Cursor, or Cline, focusing on in-editor coding assistance.
  Choose the context that best matches the type of integration you are using.

When launching Serena, specify the context using `--context <context-name>`.
Note that for cases where parameter lists are specified (e.g. Claude Desktop), you must add two parameters to the list.

If you are using a local server (such as Llama.cpp) which requires you to use OpenAI-compatible tool descriptions, use context `oaicompat-agent` instead of `agent`.

#### Modes

Modes further refine Serena's behavior for specific types of tasks or interaction styles. Multiple modes can be active simultaneously, allowing you to combine their effects. Modes influence the system prompt and can also alter the set of available tools by excluding certain ones.

Examples of built-in modes include:

* `planning`: Focuses Serena on planning and analysis tasks.
* `editing`: Optimizes Serena for direct code modification tasks.
* `interactive`: Suitable for a conversational, back-and-forth interaction style.
* `one-shot`: Configures Serena for tasks that should be completed in a single response, often used with `planning` for generating reports or initial plans.
* `no-onboarding`: Skips the initial onboarding process if it's not needed for a particular session.
* `onboarding`: (Usually triggered automatically) Focuses on the project onboarding process.

Modes can be set at startup (similar to contexts) but can also be _switched dynamically_ during a session. You can instruct the LLM to use the `switch_modes` tool to activate a different set of modes (e.g., "switch to planning and one-shot modes").

When launching Serena, specify modes using `--mode <mode-name>`; multiple modes can be specified, e.g. `--mode planning --mode no-onboarding`.

:warning: **Mode Compatibility**: While you can combine modes, some may be semantically incompatible (e.g., `interactive` and `one-shot`). Serena currently does not prevent incompatible combinations; it is up to the user to choose sensible mode configurations.

#### Customization

You can create your own contexts and modes to precisely tailor Serena to your needs in two ways:

* You can use Serena's CLI to manage modes and contexts. Check out

    ```shell
    uvx --from git+https://github.com/oraios/serena serena mode --help
    ```

  and

    ```shell
    uvx --from git+https://github.com/oraios/serena serena context --help
    ```

  _NOTE_: Custom contexts/modes are simply YAML files in `<home>/.serena`, they are automatically registered and available for use by their name (filename without the `.yml` extension). If you don't want to use Serena's CLI, you can create and manage them in any way you see fit.
* **Using external YAML files**: When starting Serena, you can also provide an absolute path to a custom `.yml` file for a context or mode.

This customization allows for deep integration and adaptation of Serena to specific project requirements or personal preferences.

### Onboarding and Memories

By default, Serena will perform an **onboarding process** when
it is started for the first time for a project.
The goal of the onboarding is for Serena to get familiar with the project
and to store memories, which it can then draw upon in future interactions.
If an LLM should fail to complete the onboarding and does not actually write the
respective memories to disk, you may need to ask it to do so explicitly.

The onboarding will usually read a lot of content from the project, thus filling
up the context. It can therefore be advisable to switch to another conversation
once the onboarding is complete.
After the onboarding, we recommend that you have a quick look at the memories and,
if necessary, edit them or add additional ones.

**Memories** are files stored in `.serena/memories/` in the project directory,
which the agent can choose to read in subsequent interactions.
Feel free to read and adjust them as needed; you can also add new ones manually.
Every file in the `.serena/memories/` directory is a memory file.
Whenever Serena starts working on a project, the list of memories is
provided, and the agent can decide to read them.
We found that memories can significantly improve the user experience with Serena.

### Prepare Your Project

#### Structure Your Codebase

Serena uses the code structure for finding, reading and editing code. This means that it will
work well with well-structured code but may perform poorly on fully unstructured one (like a "God class"
with enormous, non-modular functions).
Furthermore, for languages that are not statically typed, type annotations are highly beneficial.

#### Start from a Clean State

It is best to start a code generation task from a clean git state. Not only will
this make it easier for you to inspect the changes, but also the model itself will
have a chance of seeing what it has changed by calling `git diff` and thereby
correct itself or continue working in a followup conversation if needed.

:warning: **Important**: since Serena will write to files using the system-native line endings
and it might want to look at the git diff, it is important to
set `git config core.autocrlf` to `true` on Windows.
With `git config core.autocrlf` set to `false` on Windows, you may end up with huge diffs
only due to line endings. It is generally a good idea to globally enable this git setting on Windows:

```shell
git config --global core.autocrlf true
```

#### Logging, Linting, and Automated Tests

Serena can successfully complete tasks in an _agent loop_, where it iteratively
acquires information, performs actions, and reflects on the results.
However, Serena cannot use a debugger; it must rely on the results of program executions,
linting results, and test results to assess the correctness of its actions.
Therefore, software that is designed to meaningful interpretable outputs (e.g. log messages)
and that has a good test coverage is much easier to work with for Serena.

We generally recommend to start an editing task from a state where all linting checks and tests pass.

### Prompting Strategies

We found that it is often a good idea to spend some time conceptualizing and planning a task
before actually implementing it, especially for non-trivial task. This helps both in achieving
better results and in increasing the feeling of control and staying in the loop. You can
make a detailed plan in one session, where Serena may read a lot of your code to build up the context,
and then continue with the implementation in another (potentially after creating suitable memories).

### Running Out of Context

For long and complicated tasks, or tasks where Serena has read a lot of content, you
may come close to the limits of context tokens. In that case, it is often a good idea to continue
in a new conversation. Serena has a dedicated tool to create a summary of the current state
of the progress and all relevant info for continuing it. You can request to create this summary and
write it to a memory. Then, in a new conversation, you can just ask Serena to read the memory and
continue with the task. In our experience, this worked really well. On the up-side, since in a
single session there is no summarization involved, Serena does not usually get lost (unlike some
other agents that summarize under the hood), and it is also instructed to occasionally check whether
it's on the right track.

Moreover, Serena is instructed to be frugal with context
(e.g., to not read bodies of code symbols unnecessarily),
but we found that Claude is not always very good in being frugal (Gemini seemed better at it).
You can explicitly instruct it to not read the bodies if you know that it's not needed.

### Serena's Logs: The Dashboard and GUI Tool

Serena provides two convenient ways of accessing the logs of the current session:

* via the **web-based dashboard** (enabled by default)

  This is supported on all platforms.
  By default, it will be accessible at `http://localhost:24282/dashboard/index.html`,
  but a higher port may be used if the default port is unavailable/multiple instances are running.

* via the **GUI tool** (disabled by default)

  This is mainly supported on Windows, but it may also work on Linux; macOS is unsupported.

Both can be enabled, configured or disabled in Serena's configuration file (`serena_config.yml`, see above).
If enabled, they will automatically be opened as soon as the Serena agent/MCP server is started.
The web dashboard will display usage statistics of Serena's tools if you set  `record_tool_usage_stats: True` in your config.

In addition to viewing logs, both tools allow to shut down the Serena agent.
This function is provided, because clients like Claude Desktop may fail to terminate the MCP server subprocess
when they themselves are closed.

### Serena and GIT worktrees
[git-worktree](https://git-scm.com/docs/git-worktree) can be an excellent way to parallelize your work. More on this in [Anthropic: Run parallel Claude Code sessions with Git worktrees](https://docs.claude.com/en/docs/claude-code/common-workflows#run-parallel-claude-code-sessions-with-git-worktrees).

When it comes to serena AND git-worktree AND larger projects (that take longer to index), the recommended way is to COPY your `$ORIG_PROJECT/.serena/cache` to `$GIT_WORKTREE/.serena/cache`. After you have performed pre-indexing of your project described in [Project Activation & Indexing](#project-activation--indexing) section. To avoid having to re-index per each git work tree that you create. 
