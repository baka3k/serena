from agno.tools import Function, Toolkit

from serena.agent import SerenaAgent, Tool


class SerenaAgnoToolkit(Toolkit):
    def __init__(self, serena_agent: SerenaAgent):
        super().__init__("Serena")
        for tool in serena_agent.tools.values():
            self.functions[tool.get_name()] = self._create_agno_function(tool)

    @staticmethod
    def _create_agno_function(tool: Tool) -> Function:
        def entrypoint(**kwargs):  # type: ignore
            tool.apply_ex(log_call=True, catch_exceptions=True, **kwargs)

        function = Function.from_callable(tool.get_apply_fn())
        function.name = tool.get_name()
        function.entrypoint = entrypoint
        function.show_result = True
        return function
