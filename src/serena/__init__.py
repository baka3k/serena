__version__ = "2025-05-21"


def serena_version() -> str:
    """
    :return: the version of the package, including git status if available.
    """
    version = __version__
    try:
        from sensai.util.git import git_status
        from sensai.util.logging import LoggingDisabledContext

        with LoggingDisabledContext():
            git_status = git_status()
        version += f"-{git_status.commit[:8]}"
        if not git_status.is_clean:
            version += "-dirty"
    except:
        pass
    return version
