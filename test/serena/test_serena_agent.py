import json
import os
import time

import pytest

from multilspy.multilspy_config import Language
from serena.agent import FindReferencingSymbolsTool, FindSymbolTool, ProjectConfig, SerenaAgent, SerenaConfig
from test.conftest import LanguageParamRequest, get_repo_path


@pytest.fixture
def serena_config():
    return SerenaConfig()


@pytest.fixture
def serena_agent(request: LanguageParamRequest, serena_config) -> SerenaAgent:
    language = Language(request.param)
    repo = get_repo_path(language)
    project_config = ProjectConfig(project_name=os.path.basename(str(repo)), language=language, project_root=str(repo))
    return SerenaAgent(project_config=project_config, serena_config=serena_config)


class TestSerenaAgent:
    @pytest.mark.parametrize(
        "serena_agent,symbol_name,expected_kind,expected_file",
        [
            (Language.PYTHON, "User", "Class", "models.py"),
            (Language.GO, "Helper", "Function", "main.go"),
            (Language.JAVA, "Model", "Class", "Model.java"),
            (Language.RUST, "add", "Function", "lib.rs"),
            (Language.TYPESCRIPT, "DemoClass", "Class", "index.ts"),
        ],
        indirect=["serena_agent"],
    )
    def test_find_symbol(self, serena_agent, symbol_name, expected_kind, expected_file):
        agent = serena_agent
        find_symbol_tool = agent.get_tool(FindSymbolTool)
        result = find_symbol_tool.apply(symbol_name)
        symbols = json.loads(result)
        assert any(
            symbol_name in s["name"] and expected_kind.lower() in s["kind"].lower() and expected_file in s["location"]["relative_path"]
            for s in symbols
        ), f"Expected to find {symbol_name} ({expected_kind}) in {expected_file} for {agent.project_config.language.name}"

    @pytest.mark.parametrize(
        "serena_agent,symbol_name,def_file,ref_file",
        [
            (Language.PYTHON, "User", "test_repo/models.py", "test_repo/services.py"),
            (Language.GO, "Helper", "main.go", "main.go"),
            (Language.JAVA, "Model", "src/main/java/test_repo/Model.java", "src/main/java/test_repo/Main.java"),
            (Language.RUST, "add", "src/lib.rs", "src/main.rs"),
            (Language.TYPESCRIPT, "helperFunction", "index.ts", "use_helper.ts"),
        ],
        indirect=["serena_agent"],
    )
    def test_find_symbol_references(self, serena_agent: SerenaAgent, symbol_name: str, def_file: str, ref_file: str) -> None:
        agent = serena_agent
        # Find the symbol location first
        find_symbol_tool = agent.get_tool(FindSymbolTool)
        result = find_symbol_tool.apply(symbol_name, within_relative_path=def_file)
        time.sleep(1)
        symbols = json.loads(result)
        # Find the definition location
        def_symbol = symbols[0]
        loc = def_symbol["location"]
        # sel_start = def_symbol["location"]["selectionRange"]["start"]
        # Now find references
        find_refs_tool = agent.get_tool(FindReferencingSymbolsTool)
        result = find_refs_tool.apply(def_file, loc["line"], loc["column"])
        refs = json.loads(result)
        assert any(
            ref["location"]["relative_path"] == ref_file for ref in refs
        ), f"Expected to find reference to {symbol_name} in {ref_file} for {agent.project_config.language.name}. refs={refs}"
