import argparse
import os
from logging import Logger
from pathlib import Path

from agno.agent.agent import Agent
from agno.memory.agent import AgentMemory
from agno.models.anthropic.claude import Claude
from agno.models.google.gemini import Gemini
from agno.playground.playground import Playground
from agno.playground.serve import serve_playground_app
from agno.storage.sqlite import SqliteStorage
from dotenv import load_dotenv
from sensai.util import logging
from sensai.util.helper import mark_used

from serena.agent import SerenaAgent
from serena.agno import SerenaAgnoToolkit

mark_used(Gemini, Claude)

os.chdir(Path(__file__).parent)
Logger.root.setLevel(logging.INFO)

load_dotenv()


parser = argparse.ArgumentParser(description="Serena coding assistant")
parser.add_argument("--project-file", required=True, help="Path to the project file, either absolute or relative to the root directory")
args = parser.parse_args()

project_file = Path(args.project_file).resolve()
# If project file path is relative, make it absolute by joining with project root
if not project_file.is_absolute():
    # Get the project root directory (parent of scripts directory)
    project_root = Path(__file__).parent.parent
    project_file = project_root / args.project_file

# Ensure the path is normalized and absolute
project_file = project_file.resolve()


serena_agent = SerenaAgent(str(project_file), start_language_server=True)

# Even though we don't want to keep history between sessions,
# for the agno ui to work as a conversation, we a persistent storage on disk
# This storage should be deleted between sessions.
# Note that this might collide with custom options for the agent, like adding vector-search based tools.
# See here for an explanation: https://www.reddit.com/r/agno/comments/1jk6qea/regarding_the_built_in_memory/
sql_db_path = Path("tmp") / "agent_storage.db"
sql_db_path.parent.mkdir(exist_ok=True)
# delete the db file if it exists
if sql_db_path.exists():
    sql_db_path.unlink()

# model = Claude(id="claude-3-7-sonnet-20250219")
# Or use any other model supported by agno, e.g. the newest Gemini:
model = Gemini(id="gemini-2.5-pro-exp-03-25")

agno_agent = Agent(
    name="Serena",
    model=model,
    # See explanation above on why storage is needed
    storage=SqliteStorage(table_name="serena_agent_sessions", db_file=str(sql_db_path)),
    description="A fully-featured coding assistant",
    tools=[SerenaAgnoToolkit(serena_agent)],  # type: ignore
    # The tool calls will be shown in the UI anyway since whether to show them is configurable per tool
    # To see detailed logs, you should use the serena logger (configure it in the project file path)
    show_tool_calls=False,
    markdown=True,
    system_message=serena_agent.prompt_factory.create_system_prompt(),
    telemetry=False,
    memory=AgentMemory(),
    add_history_to_messages=True,
    num_history_responses=100,  # you might want to adjust this (expense vs. history awareness)
)

app = Playground(agents=[agno_agent]).get_app()


if __name__ == "__main__":
    serve_playground_app("agno_agent:app", reload=False)
