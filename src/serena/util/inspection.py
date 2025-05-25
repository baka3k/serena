import os
from collections.abc import Generator
from typing import TypeVar

from multilspy.multilspy_config import Language
from serena.util.file_system import scan_directory

T = TypeVar("T")


def iter_subclasses(cls: type[T], recursive: bool = True) -> Generator[type[T], None, None]:
    """Iterate over all subclasses of a class. If recursive is True, also iterate over all subclasses of all subclasses."""
    for subclass in cls.__subclasses__():
        yield subclass
        if recursive:
            yield from iter_subclasses(subclass, recursive)


def determine_programming_language_composition(repo_path: str) -> dict[str, float]:
    """
    Determine the programming language composition of a repository.

    :param repo_path: Path to the repository to analyze
    :return: Dictionary mapping language names to percentages of files matching each language
    """
    if not os.path.exists(repo_path):
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    if not os.path.isdir(repo_path):
        raise ValueError(f"Repository path is not a directory: {repo_path}")

    # Scan all files in the repository recursively
    _, all_files = scan_directory(
        path=repo_path,
        recursive=True,
        relative_to=repo_path,
        is_ignored_dir=lambda dirname: os.path.basename(dirname).startswith("."),  # Ignore hidden directories
        is_ignored_file=lambda filename: os.path.basename(filename).startswith("."),  # Ignore hidden files
    )

    if not all_files:
        return {}

    # Count files for each language
    language_counts: dict[str, int] = {}
    total_files = len(all_files)

    for language in Language:
        matcher = language.get_source_fn_matcher()
        count = 0

        for file_path in all_files:
            # Use just the filename for matching, not the full path
            filename = os.path.basename(file_path)
            if matcher.is_relevant_filename(filename):
                count += 1

        if count > 0:
            language_counts[str(language)] = count

    # Convert counts to percentages
    language_percentages: dict[str, float] = {}
    for language_name, count in language_counts.items():
        percentage = (count / total_files) * 100
        language_percentages[language_name] = round(percentage, 2)

    return language_percentages
