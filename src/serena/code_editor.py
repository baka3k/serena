import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Reversible
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generic, Optional, TypeVar

from serena.symbol import AbstractSymbol, PositionInFile, Symbol, SymbolManager
from solidlsp import SolidLanguageServer
from solidlsp.ls import LSPFileBuffer

if TYPE_CHECKING:
    from .agent import SerenaAgent


TSymbol = TypeVar("TSymbol", bound=AbstractSymbol)


class CodeEditor(Generic[TSymbol], ABC):
    def __init__(self, project_root: str, agent: Optional["SerenaAgent"] = None) -> None:
        self.project_root = project_root
        self.agent = agent

    class EditedFile(ABC):
        @abstractmethod
        def get_contents(self) -> str:
            """
            :return: the contents of the file.
            """

        @abstractmethod
        def delete_text_between_positions(self, start_pos: PositionInFile, end_pos: PositionInFile) -> None:
            pass

        @abstractmethod
        def insert_text_at_position(self, pos: PositionInFile, text: str) -> None:
            pass

    @contextmanager
    def _open_file_context(self, relative_path: str) -> Iterator["CodeEditor.EditedFile"]:
        """
        Context manager for opening a file
        """
        raise NotImplementedError("This method must be overridden for each subclass")

    @contextmanager
    def _edited_file_context(self, relative_path: str) -> Iterator["CodeEditor.EditedFile"]:
        """
        Context manager for editing a file.
        """
        with self._open_file_context(relative_path) as edited_file:
            yield edited_file
            # save the file
            abs_path = os.path.join(self.project_root, relative_path)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(edited_file.get_contents())
            # notify agent (if provided)
            if self.agent is not None:
                self.agent.mark_file_modified(relative_path)

    @abstractmethod
    def _find_unique_symbol(self, name_path: str, relative_file_path: str) -> TSymbol:
        """
        Finds the unique symbol with the given name in the given file.
        If no such symbol exists, raises a ValueError.

        :param name_path: the name path
        :param relative_file_path: the relative path of the file in which to search for the symbol.
        :return: the unique symbol
        """

    def replace_body(self, name_path: str, relative_file_path: str, body: str) -> None:
        """
        Replaces the body of the symbol with the given name_path in the given file.

        :param name_path: the name path of the symbol to replace.
        :param relative_file_path: the relative path of the file in which the symbol is defined.
        :param body: the new body
        """
        symbol = self._find_unique_symbol(name_path, relative_file_path)
        start_pos = symbol.get_body_start_position_or_raise()
        end_pos = symbol.get_body_end_position_or_raise()

        with self._edited_file_context(relative_file_path) as edited_file:
            # make sure the replacement adds no additional newlines (before or after) - all newlines
            # and whitespace before/after should remain the same, so we strip it entirely
            body = body.strip()

            edited_file.delete_text_between_positions(start_pos, end_pos)
            edited_file.insert_text_at_position(start_pos, body)

    @staticmethod
    def _count_leading_newlines(text: Iterable) -> int:
        cnt = 0
        for c in text:
            if c == "\n":
                cnt += 1
            elif c == "\r":
                continue
            else:
                break
        return cnt

    @classmethod
    def _count_trailing_newlines(cls, text: Reversible) -> int:
        return cls._count_leading_newlines(reversed(text))

    def insert_after_symbol(self, name_path: str, relative_file_path: str, body: str) -> None:
        """
        Inserts content after the symbol with the given name in the given file.
        """
        symbol = self._find_unique_symbol(name_path, relative_file_path)

        # make sure body always ends with at least one newline
        if not body.endswith("\n"):
            body += "\n"

        pos = symbol.get_body_end_position_or_raise()

        # start at the beginning of the next line
        col = 0
        line = pos.line + 1

        # make sure a suitable number of leading empty lines is used (at least 0/1 depending on the symbol type,
        # otherwise as many as the caller wanted to insert)
        original_leading_newlines = self._count_leading_newlines(body)
        body = body.lstrip("\r\n")
        min_empty_lines = 0
        if symbol.is_neighbouring_definition_separated_by_empty_line():
            min_empty_lines = 1
        num_leading_empty_lines = max(min_empty_lines, original_leading_newlines)
        if num_leading_empty_lines:
            body = ("\n" * num_leading_empty_lines) + body

        # make sure the one line break succeeding the original symbol, which we repurposed as prefix via
        # `line += 1`, is replaced
        body = body.rstrip("\r\n") + "\n"

        with self._edited_file_context(relative_file_path) as edited_file:
            edited_file.insert_text_at_position(PositionInFile(line, col), body)

    def insert_before_symbol(self, name_path: str, relative_file_path: str, body: str) -> None:
        """
        Inserts content before the symbol with the given name in the given file.
        """
        symbol = self._find_unique_symbol(name_path, relative_file_path)
        symbol_start_pos = symbol.get_body_start_position_or_raise()

        # insert position is the start of line where the symbol is defined
        line = symbol_start_pos.line
        col = 0

        original_trailing_empty_lines = self._count_trailing_newlines(body) - 1

        # ensure eol is present at end
        body = body.rstrip() + "\n"

        # add suitable number of trailing empty lines after the body (at least 0/1 depending on the symbol type,
        # otherwise as many as the caller wanted to insert)
        min_trailing_empty_lines = 0
        if symbol.is_neighbouring_definition_separated_by_empty_line():
            min_trailing_empty_lines = 1
        num_trailing_newlines = max(min_trailing_empty_lines, original_trailing_empty_lines)
        body += "\n" * num_trailing_newlines

        # apply edit
        with self._edited_file_context(relative_file_path) as edited_file:
            edited_file.insert_text_at_position(PositionInFile(line=line, col=col), body)

    def insert_at_line(self, relative_path: str, line: int, content: str) -> None:
        """
        Inserts content at the given line in the given file.

        :param relative_path: the relative path of the file in which to insert content
        :param line: the 0-based index of the line to insert content at
        :param content: the content to insert
        """
        with self._edited_file_context(relative_path) as edited_file:
            edited_file.insert_text_at_position(PositionInFile(line, 0), content)

    def delete_lines(self, relative_path: str, start_line: int, end_line: int) -> None:
        """
        Deletes lines in the given file.

        :param relative_path: the relative path of the file in which to delete lines
        :param start_line: the 0-based index of the first line to delete (inclusive)
        :param end_line: the 0-based index of the last line to delete (inclusive)
        """
        start_col = 0
        end_line_for_delete = end_line + 1
        end_col = 0
        with self._edited_file_context(relative_path) as edited_file:
            start_pos = PositionInFile(line=start_line, col=start_col)
            end_pos = PositionInFile(line=end_line_for_delete, col=end_col)
            edited_file.delete_text_between_positions(start_pos, end_pos)

    def delete_symbol(self, name_path: str, relative_file_path: str) -> None:
        """
        Deletes the symbol with the given name in the given file.
        """
        symbol = self._find_unique_symbol(name_path, relative_file_path)
        start_pos = symbol.get_body_start_position_or_raise()
        end_pos = symbol.get_body_end_position_or_raise()
        with self._edited_file_context(relative_file_path) as edited_file:
            edited_file.delete_text_between_positions(start_pos, end_pos)


class LanguageServerCodeEditor(CodeEditor[Symbol]):
    def __init__(self, symbol_manager: SymbolManager, agent: Optional["SerenaAgent"] = None):
        super().__init__(project_root=symbol_manager.get_language_server().repository_root_path, agent=agent)
        self._symbol_manager = symbol_manager

    @property
    def _lang_server(self) -> SolidLanguageServer:
        return self._symbol_manager.get_language_server()

    class EditedFile(CodeEditor.EditedFile):
        def __init__(self, lang_server: SolidLanguageServer, relative_path: str, file_buffer: LSPFileBuffer):
            self._lang_server = lang_server
            self._relative_path = relative_path
            self._file_buffer = file_buffer

        def get_contents(self) -> str:
            return self._file_buffer.contents

        def delete_text_between_positions(self, start_pos: PositionInFile, end_pos: PositionInFile) -> None:
            self._lang_server.delete_text_between_positions(self._relative_path, start_pos.to_lsp_position(), end_pos.to_lsp_position())

        def insert_text_at_position(self, pos: PositionInFile, text: str) -> None:
            self._lang_server.insert_text_at_position(self._relative_path, pos.line, pos.col, text)

    @contextmanager
    def _open_file_context(self, relative_path: str) -> Iterator["CodeEditor.EditedFile"]:
        with self._lang_server.open_file(relative_path) as file_buffer:
            yield self.EditedFile(self._lang_server, relative_path, file_buffer)

    def _get_code_file_content(self, relative_path: str) -> str:
        """Get the content of a file using the language server."""
        return self._lang_server.language_server.retrieve_full_file_content(relative_path)

    def _find_unique_symbol(self, name_path: str, relative_file_path: str) -> Symbol:
        symbol_candidates = self._symbol_manager.find_by_name(name_path, within_relative_path=relative_file_path)
        if len(symbol_candidates) == 0:
            raise ValueError(f"No symbol with name {name_path} found in file {relative_file_path}")
        if len(symbol_candidates) > 1:
            raise ValueError(
                f"Found multiple {len(symbol_candidates)} symbols with name {name_path} in file {relative_file_path}. "
                "Their locations are: \n " + json.dumps([s.location.to_dict() for s in symbol_candidates], indent=2)
            )
        return symbol_candidates[0]
