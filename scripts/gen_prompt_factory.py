from sensai.util import logging

from serena.llm.multilang_prompt import MultiLangPromptTemplateCollection

log = logging.getLogger(__name__)


def main():
    coll = MultiLangPromptTemplateCollection()

    package = "src/serena"

    # collect methods to generate
    indent = "    "
    methods = []
    for mpt in coll.prompt_templates.values():
        prompt_name = mpt.name
        params = mpt.get_parameters()
        if len(params) == 0:
            params_str = ""
        else:
            params_str = ", *, " + ", ".join(params)
        methods.append(
            f"def create_{prompt_name}(self{params_str}) -> str:"
            + f"\n{indent}{indent}return self._format_prompt('{prompt_name}', locals())\n\n{indent}"
        )
    for mpl in coll.prompt_lists.values():
        prompt_name = mpl.name
        methods.append(
            f"def get_list_{prompt_name}(self) -> PromptList:" + f"\n{indent}{indent}return self._get_list('{prompt_name}')\n\n{indent}"
        )

    # write prompt factory with added methods
    with open("code_templates/prompt_factory_template.py", encoding="utf-8") as f:
        code = f.read()
    methods_str = "".join(methods)
    code = code.replace("# methods", methods_str)

    prompt_factory_module = f"{package}/llm/prompt_factory.py"
    with open(prompt_factory_module, "w", encoding="utf-8") as f:
        f.write(code)
    log.info(f"Prompt factory generated successfully in {prompt_factory_module}")


if __name__ == "__main__":
    logging.run_main(main)
