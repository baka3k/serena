# Lessons Learned

In this document we briefly collect what we have learned while developing and using Serena,  
what works well and what doesn't.

## What Worked

### Separate Tool Logic From MCP Implementation

MCP is just another protocol, one should let the details of it creep into the application logic.  
The official docs suggest using function annotations to define tools and prompts. While that may be  
useful for small projects to get going fast, it is not wise for more serious projects. In Serena,  
all tools are defined independently and then converted to instances of `MCPTool` using our `make_tool`  
function.

### Tempfiles and Snapshots for Testing of Editing Tools

We test most aspects of Serena by having a small "project" for each supported language in `tests/resources`.  
For the editing tools, which would change the code in these projects, we use tempfiles to copy over the code.  
The pretty awesome [syrupy](https://github.com/syrupy-project/syrupy) pytest plugin helped in developing  
snapshot tests.

### Dashboard and GUI for Logging

It is very useful to know what the MCP Server is doing. We collect and display logs in a GUI or a web dashboard,  
which helps a lot in seeing what's going on and in identifying any issues.

### Unrestricted Bash Tool

We know it's not particularly safe to permit unlimited shell commands outside a sandbox, but we did quite some  
evaluations and so far... nothing bad has happened. Seems like the current versions of the AI overlords rarely want to execute `sudo rm - rf /`.  
Still, we are working on a safer approach as well as better integration with sandboxing.

### Multilspy

The [multilspy](https://github.com/microsoft/multilspy/) project helped us a lot in getting started and stands at the core of Serena.  
Many more well known python implementations of language servers were subpar in code quality and design (for example, missing types).

### Developing Serena with Serena

We clearly notice that the better the tool gets, the easier it is to make it even better

## What Didn't Work

### Lifespan Handling by MCP Clients

The MCP technology is clearly very green. Even though there is a lifespan context in the MCP SDK,  
many clients, including Claude Desktop, fail to properly clean up, leaving zombie processes behind.  
We mitigate this through the GUI window and the dashboard, so the user sees whether Serena is running  
and can terminate it there.

### Cross-OS Tkinter GUI

Different OS have different limitations when it comes to starting a window or dealing with Tkinter  
installations. This was so messy to get right that we pivoted to a web-dashboard instead

### Editing Based on Line Numbers

Not only are LLMs notoriously bad in counting, but also the line numbers change after edit operations,  
and LLMs are also often too dumb to understand that they should update the line numbers information they had  
received before. We pivoted to string-matching and symbol-name based editing.