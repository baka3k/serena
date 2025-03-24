from pathlib import Path

import pytest

from multilspy.language_server import SyncLanguageServer
from multilspy.multilspy_config import Language, MultilspyConfig
from multilspy.multilspy_logger import MultilspyLogger


@pytest.fixture(scope="session")
def resources_dir() -> Path:
    """Path to the test resources directory."""
    current_dir = Path(__file__).parent
    return current_dir / "resources"


@pytest.fixture(scope="session")
def repo_path() -> Path:
    return Path(__file__).parent / "resources" / "test_repo"


@pytest.fixture(scope="session")
def language_server(repo_path: Path):
    """Create a SyncLanguageServer instance configured to use the test repository."""
    config = MultilspyConfig(code_language=Language.PYTHON)
    logger = MultilspyLogger()

    # Create a language server instance
    server = SyncLanguageServer.create(config, logger, str(repo_path))

    # Start the server
    server.start()

    try:
        yield server
    finally:
        # Ensure server is shut down
        server.stop()
