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
