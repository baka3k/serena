from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.playground import Playground, serve_playground_app
from dotenv import load_dotenv

from serena.agent import SerenaAgent
from serena.agno import SerenaAgnoToolkit

load_dotenv()

project_file_path = "../myproject.yml"
serena_agent = SerenaAgent(project_file_path)
model = Claude(id="claude-3-7-sonnet-20250219")

agno_agent = Agent(
    name="Serena",
    model=model,
    description="A fully-featured coding assistant",
    tools=[SerenaAgnoToolkit(serena_agent)],  # type: ignore
    show_tool_calls=True,
    markdown=True,
    system_message="",  # Todo
    read_tool_call_history=True,
    telemetry=False,
)


## The app object should be in the module scope so that the server can access it for hot reloading
app = Playground(agents=[agno_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("agno_agent:app", reload=True)
