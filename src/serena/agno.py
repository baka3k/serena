import logging
from typing import Any

from agno.tools import Function, Toolkit

from serena.agent import SerenaAgent, Tool

log = logging.getLogger(__name__)


class SerenaAgnoToolkit(Toolkit):
    def __init__(self, serena_agent: SerenaAgent):
        super().__init__("Serena")
        for tool in serena_agent.tools.values():
            self.functions[tool.get_name()] = self._create_agno_function(tool)
        log.info("Agno agent functions: %s", list(self.functions.keys()))

    @staticmethod
    def _create_agno_function(tool: Tool) -> Function:
        def entrypoint(**kwargs: Any) -> str:
            if "kwargs" in kwargs:
                # Agno sometimes passes a kwargs argument explicitly, so we merge it
                kwargs.update(kwargs["kwargs"])
                del kwargs["kwargs"]
            log.info(f"Calling tool {tool}")
            return tool.apply_ex(log_call=True, catch_exceptions=True, **kwargs)

        function = Function.from_callable(tool.get_apply_fn())
        function.name = tool.get_name()
        function.entrypoint = entrypoint
        function.show_result = True
        return function
