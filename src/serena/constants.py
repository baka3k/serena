from pathlib import Path

_repo_root_path = Path(__file__).parent.parent.parent

REPO_ROOT = str(_repo_root_path)
PROMPT_TEMPLATES_DIR = str(_repo_root_path / "config" / "prompt_templates")
CONTEXT_YAMLS_DIR = str(_repo_root_path / "config" / "contexts")
MODE_YAMLS_DIR = str(_repo_root_path / "config" / "modes")

DEFAULT_CONTEXT = "desktop-app"
DEFAULT_MODES = ("interactive", "editing")
