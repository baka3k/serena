"""
This script demonstrates how to use Serena's tools locally, useful
for testing or development. Here the tools will be operation the serena repo itself.
"""

from pprint import pprint

from serena.agent import *
from serena.constants import REPO_ROOT


@dataclass
class InMemorySerenaConfig(SerenaConfigBase):
    """
    In-memory implementation of Serena configuration with the GUI disabled.
    """

    gui_log_window_enabled: bool = False
    web_dashboard: bool = False


if __name__ == "__main__":
    agent = SerenaAgent(project=REPO_ROOT, serena_config=InMemorySerenaConfig())

    # apply a tool
    find_refs_tool = agent.get_tool(FindReferencingSymbolsTool)
    find_file_tool = agent.get_tool(FindFileTool)
    search_pattern_tool = agent.get_tool(SearchForPatternTool)

    result = agent.execute_task(
        lambda: search_pattern_tool.apply("\n[^\n]*?Pyright", restrict_search_to_code_files=False, relative_path="src/solidlsp/ls.py")
    )
    pprint(json.loads(result))
