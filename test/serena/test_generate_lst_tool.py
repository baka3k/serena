import json
import logging
import os

import pytest

from serena.agent import SerenaAgent
from serena.config.serena_config import RegisteredProject, SerenaConfig
from serena.project import Project
from serena.tools import GenerateLosslessSemanticTreeTool
from solidlsp.ls_config import Language
from test.conftest import get_repo_path


@pytest.fixture(scope="module")
def python_agent() -> SerenaAgent:
    """Spin up a SerenaAgent bound to the python reference repo."""
    repo_path = str(get_repo_path(Language.PYTHON))
    project = Project.load(repo_path)
    config = SerenaConfig(gui_log_window_enabled=False, web_dashboard=False, log_level=logging.ERROR)
    config.projects = [RegisteredProject.from_project_instance(project)]
    agent = SerenaAgent(project=project.project_name, serena_config=config)
    yield agent
    active_project = agent.get_active_project_or_raise()
    active_project.shutdown()


class TestGenerateLosslessSemanticTreeTool:
    @pytest.mark.python
    def test_lst_includes_source_and_children(self, python_agent: SerenaAgent):
        tool = python_agent.get_tool(GenerateLosslessSemanticTreeTool)
        relative_path = os.path.join("test_repo", "models.py")

        result = json.loads(tool.apply_ex(relative_path=relative_path, name_path="User"))
        root = result["root"]

        assert root["name"] == "User"
        assert root["kind"].lower() == "class"
        assert "source" in root and "class User" in root["source"]
        assert any(child["kind"].lower() == "function" or child["kind"].lower() == "method" for child in root["children"])

    @pytest.mark.python
    def test_lst_respects_depth_and_source_flag(self, python_agent: SerenaAgent):
        tool = python_agent.get_tool(GenerateLosslessSemanticTreeTool)
        relative_path = os.path.join("test_repo", "models.py")

        result = json.loads(
            tool.apply_ex(relative_path=relative_path, name_path="User", include_source=False, max_depth=0, max_answer_chars=10_000)
        )
        root = result["root"]

        assert root["children"] == []
        assert "source" not in root
