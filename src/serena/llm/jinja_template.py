from typing import Any

import jinja2
import jinja2.meta
import jinja2.nodes
import jinja2.visitor

from serena.util.class_decorators import singleton


@singleton
class JinjaEnvProvider:
    def __init__(self) -> None:
        self._env: jinja2.Environment | None = None

    def get_env(self) -> jinja2.Environment:
        if self._env is None:
            self._env = jinja2.Environment()
        return self._env


class JinjaTemplate:
    def __init__(self, template_string: str) -> None:
        self._template_string = template_string
        self._template = JinjaEnvProvider().get_env().from_string(self._template_string)

    def render(self, **kwargs: Any) -> str:
        return self._template.render(**kwargs)

    def get_parameters(self) -> set[str]:
        parsed_content = self._template.environment.parse(self._template_string)
        return jinja2.meta.find_undeclared_variables(parsed_content)
