from agno.tools import Function

from serena.agent import SerenaAgent, Tool


def serena_tool_to_agno_function(tool: Tool) -> Function:
    def entrypoint(*args, tool=tool, **kwargs):  # type: ignore
        tool.apply_ex(*args, log_call=True, catch_exceptions=True, **kwargs)

    function = Function.from_callable(tool.get_apply_fn())
    function.name = tool.get_name()
    function.entrypoint = entrypoint
    function.show_result = True
    return function


def get_agno_functions(serena_agent: SerenaAgent) -> list[Function]:
    return [serena_tool_to_agno_function(tool) for tool in serena_agent.tools.values()]
