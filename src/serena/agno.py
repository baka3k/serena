import logging
from typing import Any

from agno.tools.function import Function
from agno.tools.toolkit import Toolkit

from serena.agent import SerenaAgent, Tool

log = logging.getLogger(__name__)


def _patch_gemini_schema_conversion() -> None:
    """
    This fixes errors in Agno's Gemini schema conversion, which does not correctly handle
    optional types (union with None)
    """
    from agno.models.google import gemini
    from google.genai.types import Schema

    def _convert_schema(schema_dict: dict) -> Schema | None:
        schema_type = schema_dict.get("type", "")
        if isinstance(schema_type, list):
            schema_type = schema_type[0]
        schema_type = schema_type.upper()
        description = schema_dict.get("description", "")

        if schema_type == "OBJECT" and "properties" in schema_dict:
            properties = {key: _convert_schema(prop_def) for key, prop_def in schema_dict["properties"].items()}
            required = schema_dict.get("required", [])

            if properties:
                return Schema(
                    type=schema_type,
                    properties=properties,
                    required=required,
                    description=description,
                )
            else:
                return None

        elif schema_type == "ARRAY" and "items" in schema_dict:
            items = _convert_schema(schema_dict["items"])
            return Schema(type=schema_type, description=description, items=items)

        elif schema_type == "":
            if "anyOf" in schema_dict:
                relevant_sub_schemas = []
                is_optional = False
                for sub_schema in schema_dict["anyOf"]:
                    if sub_schema.get("type") == "null":
                        is_optional = True  # noqa: F841
                        continue
                    relevant_sub_schemas.append(sub_schema)
                # TODO handle is_optional (requires handling at the outer level)
                if len(relevant_sub_schemas) == 1:
                    return _convert_schema(relevant_sub_schemas[0])
                else:
                    return Schema(any_of=[_convert_schema(item) for item in schema_dict["anyOf"]], description=description)
            else:
                raise ValueError(f"Unhandled schema: {schema_dict}")
        else:
            return Schema(type=schema_type, description=description)

    gemini._convert_schema = _convert_schema


_patch_gemini_schema_conversion()


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
        function.skip_entrypoint_processing = True
        return function
