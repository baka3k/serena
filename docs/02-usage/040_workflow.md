## The Project Workflow

Serena uses a project-based workflow.
A **project** is simply a directory on your filesystem that contains code and other files
that you want Serena to work with.

Assuming that you have project you want to work with (which may initially be empty),
setting up a project with Serena typically involves the following steps:

1. **Project creation**: Configuring project settings for Serena (and indexing the project, if desired)
2. **Project activation**: Making Serena aware of the project you want to work with
3. **Onboarding**: Getting Serena familiar with the project (creating memories)
4. **Working on coding tasks**: Using Serena to help you with actual coding tasks in the project

### Project Creation & Indexing

You can create a project either  
 * implicitly, by just activating a directory as a project while already in a conversation; this will use default settings for your project (skip to the next section).
 * explicitly, using the project creation command, or

#### Explicit Project Creation

To explicitly create a project, use the following command while in the project directory:

    <serena> project generate-yml [options]

For instance, when using `uvx`, run

    uvx --from git+https://github.com/oraios/serena serena project generate-yml [options]

 * For an empty project, you will need to specify the programming language
   (e.g., `--language python`). 
 * For an existing project, the main programming language will be detected automatically,
   but you can choose to explicitly specify multiple languages by passing the `--language` parameter
   multiple times (e.g. `--language python --language typescript`).

After creation, you can adjust the project settings in the generated `.serena/project.yml` file.

#### Indexing

Especially for larger project, it is advisable to index the project after creation (in order to avoid
delays during MCP server startup or the first tool application):

While in the project directory, run this command:
   
    <serena> project index

Indexing has to be called only once. During regular usage, Serena will automatically update the index whenever files change.

### Project Activation
   
Project activation makes Serena aware of the project you want to work with.
You can either choose to do this
 * while in a conversation, by telling the model to activate a project, e.g.,
       
      * "Activate the project /path/to/my_project" (for first-time activation with auto-creation)
      * "Activate the project my_project"
   
   Note that this option requires the `activate_project` tool to be active (which it isn't in context `ide-assistant` where t.        

 * when the MCP server starts, by passing the project path or name as a command-line argument
   (e.g. when working on a fixed project in `ide-assistant` mode): `--project <path|name>`


### Onboarding & Memories

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