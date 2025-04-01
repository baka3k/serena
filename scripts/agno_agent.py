import os
from logging import Logger

from agno.agent.agent import Agent
from agno.memory.agent import AgentMemory
from agno.models.anthropic.claude import Claude
from agno.models.google.gemini import Gemini
from agno.playground.playground import Playground
from agno.playground.serve import serve_playground_app
from dotenv import load_dotenv
from sensai.util import logging
from sensai.util.helper import mark_used

from serena.agent import SerenaAgent
from serena.agno import SerenaAgnoToolkit

mark_used(Gemini, Claude)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
Logger.root.setLevel(logging.INFO)

load_dotenv()

project_file_path = "../myproject.yml"
serena_agent = SerenaAgent(project_file_path, start_language_server=True)

model = Claude(id="claude-3-7-sonnet-20250219")
# model = Gemini(id="gemini-2.5-pro-exp-03-25")

agno_agent = Agent(
    name="Serena",
    model=model,
    description="A fully-featured coding assistant",
    tools=[SerenaAgnoToolkit(serena_agent)],  # type: ignore
    show_tool_calls=False,
    markdown=True,
    system_message=serena_agent.prompt_factory.create_system_prompt(),
    read_tool_call_history=False,
    telemetry=False,
    memory=AgentMemory(),
    add_history_to_messages=True,
    num_history_responses=100,  # you might want to adjust this (expense vs. history awareness)
)

# The app object must be in the module scope so that the server can access it for hot reloading
app = Playground(agents=[agno_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("agno_agent:app", reload=False)
