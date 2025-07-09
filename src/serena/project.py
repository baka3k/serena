import logging
import os
from pathlib import Path
from typing import Self

import pathspec

from serena.config.serena_config import ProjectConfig
from serena.text_utils import MatchedConsecutiveLines, search_files
from serena.util.file_system import GitignoreParser, match_path
from solidlsp.ls_config import Language

log = logging.getLogger(__name__)


class Project:
    def __init__(self, project_root: str, project_config: ProjectConfig):
        self.project_root = project_root
        self.project_config = project_config

        # gather ignored paths from the project configuration and gitignore files
        ignored_paths = project_config.ignored_paths
        if len(ignored_paths) > 0:
            log.info(f"Using {len(ignored_paths)} ignored paths from the explicit project configuration.")
            log.debug(f"Ignored paths: {ignored_paths}")
        if project_config.ignore_all_files_in_gitignore:
            log.info(f"Parsing all gitignore files in {self.project_root}")
            gitignore_parser = GitignoreParser(self.project_root)
            log.info(f"Found {len(gitignore_parser.get_ignore_specs())} gitignore files.")
            for spec in gitignore_parser.get_ignore_specs():
                log.debug(f"Adding {len(spec.patterns)} patterns from {spec.file_path} to the ignored paths.")
                ignored_paths.extend(spec.patterns)

        # Set up the pathspec matcher for the ignored paths
        # for all absolute paths in ignored_paths, convert them to relative paths
        processed_patterns = []
        for pattern in set(ignored_paths):
            # Normalize separators (pathspec expects forward slashes)
            pattern = pattern.replace(os.path.sep, "/")
            processed_patterns.append(pattern)
        log.debug(f"Processing {len(processed_patterns)} ignored paths")
        self._ignore_spec = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, processed_patterns)

    @property
    def project_name(self) -> str:
        return self.project_config.project_name

    @property
    def language(self) -> Language:
        return self.project_config.language

    @classmethod
    def load(cls, project_root: str | Path, autogenerate: bool = True) -> Self:
        project_root = Path(project_root).resolve()
        if not project_root.exists():
            raise FileNotFoundError(f"Project root not found: {project_root}")
        project_config = ProjectConfig.load(project_root, autogenerate=autogenerate)
        return cls(project_root=str(project_root), project_config=project_config)

    def path_to_project_yml(self) -> str:
        return os.path.join(self.project_root, self.project_config.rel_path_to_project_yml())

    def read_file(self, relative_path: str) -> str:
        """
        Reads a file relative to the project root.

        :param relative_path: the path to the file relative to the project root
        :return: the content of the file
        """
        abs_path = Path(self.project_root) / relative_path
        if not abs_path.exists():
            raise FileNotFoundError(f"File not found: {abs_path}")
        return abs_path.read_text(encoding=self.project_config.encoding)

    def get_ignore_spec(self) -> pathspec.PathSpec:
        """Returns the pathspec matcher for the paths that were configured to be ignored through
        the multilspy config.

        This is is a subset of the full language-specific ignore spec that determines
        which files are relevant for the language server.

        This matcher is useful for operations outside of the language server,
        such as when searching for relevant non-language files in the project.
        """
        return self._ignore_spec

    def is_ignored_dirname(self, dirname: str) -> bool:
        return dirname.startswith(".")

    def is_ignored_path(self, relative_path: str, ignore_unsupported_files: bool = True) -> bool:
        """
        Determine whether a path should be ignored based on file type and ignore patterns.

        :param relative_path: Relative path to check
        :param ignore_unsupported_files: whether files that are not source files shall be ignored

        :return: whether the path should be ignored
        """
        abs_path = os.path.join(self.project_root, relative_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"File {abs_path} not found, the ignore check cannot be performed")

        # Check file extension if it's a file
        is_file = os.path.isfile(abs_path)
        if is_file and ignore_unsupported_files:
            fn_matcher = self.language.get_source_fn_matcher()
            if not fn_matcher.is_relevant_filename(abs_path):
                return True

        # Create normalized path for consistent handling
        rel_path = Path(relative_path)

        # Check each part of the path against always fulfilled ignore conditions
        dir_parts = rel_path.parts
        if is_file:
            dir_parts = dir_parts[:-1]
        for part in dir_parts:
            if not part:  # Skip empty parts (e.g., from leading '/')
                continue
            if self.is_ignored_dirname(part):
                return True

        return match_path(relative_path, self.get_ignore_spec(), root_path=self.project_root)

    def request_parsed_files(self, relative_path: str = "") -> list[str]:
        """Retrieves relative paths of all files analyzed by the Language Server.

        :param relative_path: will only retrieve files that are subpaths of this.
        """
        rel_file_paths = []
        start_path = os.path.join(self.project_root, relative_path)
        if not os.path.exists(start_path):
            raise FileNotFoundError(f"Relative path {start_path} not found.")
        if os.path.isfile(start_path):
            return [relative_path]
        else:
            for root, dirs, files in os.walk(start_path, followlinks=True):
                dirs[:] = [d for d in dirs if not self.is_ignored_path(os.path.join(root, d))]
                for file in files:
                    rel_file_path = os.path.relpath(os.path.join(root, file), start=self.project_root)
                    try:
                        if not self.is_ignored_path(rel_file_path):
                            rel_file_paths.append(rel_file_path)
                    except FileNotFoundError:
                        log.warning(
                            f"File {rel_file_path} not found (possibly due it being a symlink), skipping it in request_parsed_files",
                        )
            return rel_file_paths

    def search_files_for_pattern(
        self,
        pattern: str,
        relative_path: str = "",
        context_lines_before: int = 0,
        context_lines_after: int = 0,
        paths_include_glob: str | None = None,
        paths_exclude_glob: str | None = None,
    ) -> list[MatchedConsecutiveLines]:
        """
        Search for a pattern across all (non-ignored) source files

        :param pattern: Regular expression pattern to search for, either as a compiled Pattern or string
        :param relative_path:
        :param context_lines_before: Number of lines of context to include before each match
        :param context_lines_after: Number of lines of context to include after each match
        :param paths_include_glob: Glob pattern to filter which files to include in the search
        :param paths_exclude_glob: Glob pattern to filter which files to exclude from the search. Takes precedence over paths_include_glob.
        :return: List of matched consecutive lines with context
        """
        relative_file_paths = self.request_parsed_files(relative_path=relative_path)
        return search_files(
            relative_file_paths,
            pattern,
            root_path=self.project_root,
            context_lines_before=context_lines_before,
            context_lines_after=context_lines_after,
            paths_include_glob=paths_include_glob,
            paths_exclude_glob=paths_exclude_glob,
        )
