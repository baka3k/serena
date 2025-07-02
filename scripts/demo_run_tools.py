"""
This script demonstrates how to use Serena's tools locally, useful
for testing or development. Here the tools will be operation the serena repo itself.
"""

from pprint import pprint

from serena.agent import *
from serena.constants import REPO_ROOT

if __name__ == "__main__":
    agent = SerenaAgent(project=REPO_ROOT, serena_config=SerenaConfig(gui_log_window_enabled=False, web_dashboard=False))

    # apply a tool
    find_refs_tool = agent.get_tool(FindReferencingSymbolsTool)
    find_file_tool = agent.get_tool(FindFileTool)
    search_pattern_tool = agent.get_tool(SearchForPatternTool)

    result = agent.execute_task(
        lambda: search_pattern_tool.apply(
            r"def request_parsed_files.*?\).*?\)",
            restrict_search_to_code_files=False,
            relative_path="src/solidlsp",
            paths_include_glob="**/ls.py",
        )
    )
    pprint(json.loads(result))
