from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools

agent = Agent(
    model=Gemini(id="gemini-2.5-pro-exp-03-25"),
    description="",
    tools=[], # Add serena tools here
    show_tool_calls=True,
    markdown=True,
    context=
)
agent.print_response("Tell me about a breaking news story from New York.", stream=True)