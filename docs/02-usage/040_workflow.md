## The Project Workflow

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