"""
This script demonstrates how to use Serena's tools locally, useful
for testing or development. Here the config is pointing to the serena repo itself,
it assumes that you already have a serena_config.yml file setup.
"""

import json
from pathlib import Path
from pprint import pprint

from serena.agent import *


@dataclass
class InMemorySerenaConfig(SerenaConfigBase):
    """
    In-memory implementation of Serena configuration with the GUI disabled.
    """

    gui_log_window_enabled: bool = False
    web_dashboard: bool = False


if __name__ == "__main__":
    project_path = str(Path("test") / "resources" / "repos" / "python" / "test_repo")
    agent = SerenaAgent(project=project_path, serena_config=InMemorySerenaConfig())
    find_symbol_tool = agent.get_tool(FindSymbolTool)

    print("Finding the symbol 'VariableContainer'\n")
    pprint(json.loads(find_symbol_tool.apply("VariableContainer", within_relative_path=str(Path("test_repo") / "variables.py"))))

    # modifying with dry-run
    symbol_manager = agent.symbol_manager

    code_diff = symbol_manager.insert_after_symbol(
        "VariableContainer/modify_instance_var",
        relative_file_path=str(Path("test_repo") / "variables.py"),
        body="test test\nsecond line",
        dry_run=True,
    )
    print(f"Deleted lines: {code_diff.deleted_lines}")
    print(f"Added lines: {code_diff.added_lines}")
    print(f"Edited module: {code_diff.modified_content}")
