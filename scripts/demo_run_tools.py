"""
This script demonstrates how to use Serena's tools locally, useful
for testing or development. Here the tools will be operation the serena repo itself.
"""

import json
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

    # apply a tool
    find_symbol_tool = agent.get_tool(FindSymbolTool)
    print("Finding the symbol 'VariableContainer'\n")
    pprint(json.loads(find_symbol_tool.apply("VariableContainer", within_relative_path=str(Path("test_repo") / "variables.py"))))
