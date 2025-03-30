from agno.agent import Agent
from agno.models.base import Model
from agno.tools import Function

from serena.agent import SerenaAgent


def create_agno_agent(
    project_file_path: str, model: Model, system_prompt: str = "", markdown: bool = True, show_tool_calls: bool = True
) -> Agent:
    serena_agent = SerenaAgent(project_file_path)
    tools: list[Function] = []
    for tool in serena_agent.tools.values():

        def entrypoint(*args, tool=tool, **kwargs):  # type: ignore
            tool.apply_ex(*args, log_call=True, catch_exceptions=True, **kwargs)

        function = Function.from_callable(tool.get_apply_fn())
        function.name = tool.get_name()
        function.entrypoint = entrypoint
        tools.append(function)

    agno_agent = Agent(
        model=model,
        description="Serena",
        tools=tools,  # type: ignore
        show_tool_calls=show_tool_calls,
        markdown=markdown,
        system_message=system_prompt,
    )

    return agno_agent
