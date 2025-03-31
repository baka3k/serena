from serena.agent import SerenaAgent

if __name__ == "__main__":
    agent = SerenaAgent("myproject.yml")
    tools = {}
    for tool in agent.tools.values():
        tools[tool.get_name()] = tool
    for tool_name in sorted(tools.keys()):
        tool = tools[tool_name]
        print(f" * `{tool_name}`: {tool.get_tool_description().strip()}")
