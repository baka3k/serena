__version__ = "0.1.0-dev1"


def serena_root_path() -> str:
    from pathlib import Path

    return str(Path(__file__).parent.parent.absolute())
