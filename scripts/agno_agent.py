from agno.agent import Agent
from agno.models.google import Gemini
from agno.playground import Playground, serve_playground_app

serena_agent = Agent(
    model=Gemini(id="gemini-2.5-pro-exp-03-25"),
    description="Serena",
    tools=[],  # Add serena tools here
    show_tool_calls=True,
    markdown=True,
    system_message="",  # use PromptFactory
)

app = Playground(agents=[serena_agent]).get_app()

if __name__ == "main":
    serve_playground_app("playground:app", reload=True)
