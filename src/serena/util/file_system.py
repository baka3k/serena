import os
from collections.abc import Sequence


def scan_directory(
    path: str,
    recursive: bool = False,
    relative_to: str | None = None,
    ignored_dirs: Sequence[str] = (),
    ignored_files: Sequence[str] = (),
) -> tuple[list[str], list[str]]:
    """
    :param path: the path to scan
    :param recursive: whether to recursively scan subdirectories
    :param relative_to: the path to which the results should be relative to; if None, provide absolute paths
    :param ignored_dirs: a list of directory names or relative paths to ignore
    :param ignored_files: a list of file names or relative paths to ignore
    :return: the list of directories and files
    """
    files = []
    directories = []

    abs_path = os.path.abspath(path)
    rel_base = os.path.abspath(relative_to) if relative_to else None

    # Helper function to check if an item should be ignored
    def is_ignored(entry_path: str, ignored_items: Sequence[str]) -> bool:
        entry_name = os.path.basename(entry_path)

        # Check if name is directly in ignored list
        if entry_name in ignored_items:
            return True

        # Check if relative path matches any ignored path
        if rel_base:
            rel_path = os.path.relpath(entry_path, rel_base)
            for item in ignored_items:
                if rel_path == item or rel_path.startswith(f"{item}/"):
                    return True

        return False

    with os.scandir(abs_path) as entries:
        for entry in entries:
            entry_path = entry.path

            if rel_base:
                result_path = os.path.relpath(entry_path, rel_base)
            else:
                result_path = entry_path

            if entry.is_file():
                if not is_ignored(entry_path, ignored_files):
                    files.append(result_path)
            elif entry.is_dir():
                if not is_ignored(entry_path, ignored_dirs):
                    directories.append(result_path)
                    if recursive:
                        sub_dirs, sub_files = scan_directory(
                            entry_path, recursive=True, relative_to=relative_to, ignored_dirs=ignored_dirs, ignored_files=ignored_files
                        )
                        files.extend(sub_files)
                        directories.extend(sub_dirs)

    # Note: I've swapped the return order to match your function signature
    return directories, files
