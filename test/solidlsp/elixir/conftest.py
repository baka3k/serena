"""
Elixir-specific test configuration and fixtures.
"""
import os
import subprocess
import pytest
from pathlib import Path


def ensure_elixir_test_repo_compiled(repo_path: str) -> None:
    """Ensure the Elixir test repository is compiled for optimal Next LS performance during testing.
    
    Next LS requires the project to be fully compiled and indexed before providing
    complete references and symbol resolution. This function ensures the test repository
    is compiled before starting the language server for tests.
    
    In production environments, users typically have their code already compiled.
    
    Args:
        repo_path: Path to the Elixir project root directory
    """
    # Check if this looks like an Elixir project
    mix_file = os.path.join(repo_path, "mix.exs")
    if not os.path.exists(mix_file):
        return
        
    try:
        print(f"Compiling Elixir test repository for optimal Next LS performance...")
        result = subprocess.run(
            ["mix", "compile"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout for compilation
        )
        
        if result.returncode == 0:
            print(f"Elixir test repository compiled successfully in {repo_path}")
        else:
            print(f"Elixir test compilation completed with warnings/errors: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("Warning: Elixir test compilation timed out after 60 seconds")
    except FileNotFoundError:
        print("Warning: 'mix' command not found - Elixir test repository may not be compiled")
    except Exception as e:
        print(f"Warning: Failed to compile Elixir test repository: {e}")


@pytest.fixture(scope="session", autouse=True)
def setup_elixir_test_environment():
    """Automatically ensure Elixir test environment is ready for all Elixir tests.
    
    This fixture runs once per test session and automatically compiles the Elixir
    test repository before any Elixir tests run. It uses autouse=True so it runs
    automatically without needing to be explicitly requested by tests.
    """
    # Get the test repo path relative to this conftest.py file
    test_repo_path = Path(__file__).parent.parent.parent / "resources" / "repos" / "elixir" / "test_repo"
    ensure_elixir_test_repo_compiled(str(test_repo_path))
    return str(test_repo_path)


@pytest.fixture(scope="session")
def elixir_test_repo_path(setup_elixir_test_environment):
    """Get the path to the compiled Elixir test repository.
    
    This fixture depends on setup_elixir_test_environment to ensure compilation
    has completed before returning the path.
    """
    return setup_elixir_test_environment 