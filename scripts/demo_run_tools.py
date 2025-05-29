"""
This script demonstrates how to use Serena's tools locally, useful
for testing or development. Here the tools will be operation the serena repo itself.
"""

import json
from pprint import pprint

from serena.agent import *
from serena.constants import REPO_ROOT

if __name__ == "__main__":
    agent = SerenaAgent(project=REPO_ROOT)
    overview_tool = agent.get_tool(GetSymbolsOverviewTool)
    find_symbol_tool = agent.get_tool(FindSymbolTool)

    print("Getting an overview of the util package\n")
    pprint(json.loads(overview_tool.apply("src/serena/util")))

    print("\n\n")

    print("Finding the symbol 'SerenaAgent'\n")
    pprint(json.loads(find_symbol_tool.apply("SerenaAgent")))
