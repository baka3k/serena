"""
Elixir-specific test configuration and fixtures.
"""
import os
import subprocess
import pytest
from pathlib import Path


def ensure_elixir_test_repo_compiled(repo_path: str) -> None:
    """Ensure the Elixir test repository dependencies are installed and project is compiled.
    
    Next LS requires the project to be fully compiled and indexed before providing
    complete references and symbol resolution. This function:
    1. Installs dependencies via 'mix deps.get'
    2. Compiles the project via 'mix compile'
    
    This is essential in CI environments where dependencies aren't pre-installed.
    
    Args:
        repo_path: Path to the Elixir project root directory
    """
    # Check if this looks like an Elixir project
    mix_file = os.path.join(repo_path, "mix.exs")
    if not os.path.exists(mix_file):
        return
        
    try:
        print(f"Installing dependencies and compiling Elixir test repository for optimal Next LS performance...")
        
        # First, install dependencies
        deps_result = subprocess.run(
            ["mix", "deps.get"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout for dependency installation
        )
        
        if deps_result.returncode != 0:
            print(f"Warning: Failed to install dependencies: {deps_result.stderr}")
            # Continue anyway - some projects might not have dependencies
        
        # Then compile the project
        compile_result = subprocess.run(
            ["mix", "compile"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout for compilation
        )
        
        if compile_result.returncode == 0:
            print(f"Elixir test repository compiled successfully in {repo_path}")
        else:
            print(f"Elixir test compilation completed with warnings/errors: {compile_result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("Warning: Elixir dependency installation or compilation timed out after 60 seconds")
    except FileNotFoundError:
        print("Warning: 'mix' command not found - Elixir test repository may not be compiled")
    except Exception as e:
        print(f"Warning: Failed to prepare Elixir test repository: {e}")


@pytest.fixture(scope="session", autouse=True)
def setup_elixir_test_environment():
    """Automatically prepare Elixir test environment for all Elixir tests.
    
    This fixture runs once per test session and automatically:
    1. Installs dependencies via 'mix deps.get'
    2. Compiles the Elixir test repository via 'mix compile'
    
    It uses autouse=True so it runs automatically without needing to be explicitly 
    requested by tests. This ensures Next LS has a fully prepared project to work with.
    """
    # Get the test repo path relative to this conftest.py file
    test_repo_path = Path(__file__).parent.parent.parent / "resources" / "repos" / "elixir" / "test_repo"
    ensure_elixir_test_repo_compiled(str(test_repo_path))
    return str(test_repo_path)


@pytest.fixture(scope="session")
def elixir_test_repo_path(setup_elixir_test_environment):
    """Get the path to the prepared Elixir test repository.
    
    This fixture depends on setup_elixir_test_environment to ensure dependencies
    are installed and compilation has completed before returning the path.
    """
    return setup_elixir_test_environment 