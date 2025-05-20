import logging
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from copy import copy
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Self

from sensai.util.string import ToStringMixin

from multilspy import SyncLanguageServer
from multilspy.multilspy_types import Position, SymbolKind, UnifiedSymbolInformation

if TYPE_CHECKING:
    from .agent import SerenaAgent

log = logging.getLogger(__name__)


@dataclass
class SymbolLocation:
    """
    Represents the (start) location of a symbol identifier
    """

    relative_path: str | None
    """
    the relative path of the file containing the symbol; if None, the symbol is defined outside of the project's scope
    """
    line: int | None
    """
    the line number in which the symbol identifier is defined (if the symbol is a function, class, etc.);
    may be None for some types of symbols (e.g. SymbolKind.File)
    """
    column: int | None
    """
    the column number in which the symbol identifier is defined (if the symbol is a function, class, etc.);
    may be None for some types of symbols (e.g. SymbolKind.File)
    """

    def __post_init__(self) -> None:
        if self.relative_path is not None:
            self.relative_path = self.relative_path.replace("/", os.path.sep)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def has_position_in_file(self) -> bool:
        return self.relative_path is not None and self.line is not None and self.column is not None


class Symbol(ToStringMixin):
    _QAULNAME_SEPARATOR = "/"

    @staticmethod
    def match_against_qualname(
        name_pattern: str,
        qual_name_parts: list[str],
        substring_matching: bool,
    ) -> bool:
        """
        Checks if a given name/pattern matches a symbol's qualified name parts.

        :param name_pattern: The name or pattern to match. Can be a simple name
            (e.g., "my_func") or a qualified name pattern
            (e.g., "MyClass/my_method", "MyClass/").
        :param qual_name_parts: A list of strings representing the parts of the
            a qualified name.
        :param substring_matching: If True, allows substring matching for the relevant part(s).
            - For simple names, the whole `name_pattern` is checked as a substring
                of the symbol's simple name (`qual_name_parts[-1]`).
            - For qualified name patterns, only the *last* part of the
                `name_to_match` pattern is checked as a substring.
                Other parts must match exactly.
        :return: True if the name matches, False otherwise.
        """
        assert name_pattern, "name_to_match must not be empty"
        assert qual_name_parts, "symbol_qual_name_parts must not be empty"
        qname_separator = Symbol._QAULNAME_SEPARATOR
        is_qualified_pattern = qname_separator in name_pattern

        if not is_qualified_pattern:
            # Simple name matching
            symbol_simple_name = qual_name_parts[-1]
            if substring_matching:
                return name_pattern in symbol_simple_name
            else:
                return name_pattern == symbol_simple_name
        # Qualified name pattern matching
        # A pattern like "Lib/" or "Lib/Class/" is treated as a prefix.
        # We strip the trailing separator to get the core parts to match.
        # For example, "Lib/" becomes "Lib", "Lib/Class/" becomes "Lib/Class".
        # A pattern of just "/" becomes an empty string.
        cleaned_name_pattern = name_pattern.rstrip(qname_separator)

        # Splitting the cleaned pattern gives the segments to match.
        # - "Lib" -> ["Lib"]
        # - "Lib/Class" -> ["Lib", "Class"]
        # - "" (from "/") -> [""] (split of empty string results in list with one empty string)
        pattern_segments = cleaned_name_pattern.split(qname_separator)

        # The number of segments in the pattern must match the number of parts in the symbol's qualified name.
        # Example: pattern "Lib/Class" (2 segments) should match a symbol like `MyClass` in `MyLib` (2 qual_name_parts: ["MyLib", "MyClass"]).
        # It should not match `MyMethod` in `MyClass` in `MyLib` (3 qual_name_parts).
        # Also, pattern "Lib/" (effectively "Lib", 1 segment) should match `MyLib` (1 qual_name_part: ["MyLib"]).
        if len(pattern_segments) != len(qual_name_parts):
            return False

        # Match all segments of the pattern except the last one. These must be exact matches.
        # If pattern_segments has only one segment (e.g., "Lib", or from "Lib/", or from "/"), this loop is skipped.
        for i in range(len(pattern_segments) - 1):
            if pattern_segments[i] != qual_name_parts[i]:
                return False

        # Match the last segment of the pattern against the last part of the symbol's qualified name.
        last_pattern_segment = pattern_segments[-1]
        last_qual_name_segment = qual_name_parts[-1]

        # The `substring_matching` flag applies to this last segment comparison.
        if substring_matching:
            return last_pattern_segment in last_qual_name_segment
        else:
            return last_pattern_segment == last_qual_name_segment

    def __init__(self, symbol_root_from_ls: UnifiedSymbolInformation) -> None:
        self.symbol_root = symbol_root_from_ls

    def _tostring_includes(self) -> list[str]:
        return []

    def _tostring_additional_entries(self) -> dict[str, Any]:
        return dict(name=self.name, kind=self.kind, num_children=len(self.symbol_root["children"]))

    @property
    def name(self) -> str:
        return self.symbol_root["name"]

    @property
    def kind(self) -> str:
        return SymbolKind(self.symbol_kind).name

    @property
    def symbol_kind(self) -> SymbolKind:
        return self.symbol_root["kind"]

    @property
    def relative_path(self) -> str | None:
        location = self.symbol_root.get("location")
        if location:
            return location.get("relativePath")
        return None

    @property
    def location(self) -> SymbolLocation:
        """
        :return: the start location of the actual symbol identifier
        """
        return SymbolLocation(relative_path=self.relative_path, line=self.line, column=self.column)

    @property
    def body_start_position(self) -> Position | None:
        location = self.symbol_root.get("location")
        if location:
            range_info = location.get("range")
            if range_info:
                start_pos = range_info.get("start")
                if start_pos:
                    return start_pos
        return None

    @property
    def body_end_position(self) -> Position | None:
        location = self.symbol_root.get("location")
        if location:
            range_info = location.get("range")
            if range_info:
                end_pos = range_info.get("end")
                if end_pos:
                    return end_pos
        return None

    @property
    def line(self) -> int | None:
        if "selectionRange" in self.symbol_root:
            return self.symbol_root["selectionRange"]["start"]["line"]
        else:
            # line is expected to be undefined for some types of symbols (e.g. SymbolKind.File)
            return None

    @property
    def column(self) -> int | None:
        if "selectionRange" in self.symbol_root:
            return self.symbol_root["selectionRange"]["start"]["character"]
        else:
            # precise location is expected to be undefined for some types of symbols (e.g. SymbolKind.File)
            return None

    @property
    def body(self) -> str | None:
        return self.symbol_root.get("body")

    def get_qualified_name(self) -> str:
        """
        Get the qualified name of the symbol (e.g. "class/method/inner_function").
        """
        return self._QAULNAME_SEPARATOR.join(self.get_qualified_name_parts())

    def get_qualified_name_parts(self) -> list[str]:
        """
        Get the parts of the qualified name of the symbol (e.g. ["class", "method", "inner_function"]).
        """
        ancestors_within_file = list(self.iter_ancestors(up_to_symbol_kind=SymbolKind.File))
        ancestors_within_file.reverse()
        return [a.name for a in ancestors_within_file] + [self.name]

    def iter_children(self) -> Iterator[Self]:
        for c in self.symbol_root["children"]:
            yield self.__class__(c)

    def iter_ancestors(self, up_to_symbol_kind: SymbolKind | None = None) -> Iterator[Self]:
        """
        Iterate over all ancestors of the symbol, starting with the parent and going up to the root or
        the given symbol kind.

        :param up_to_symbol_kind: if provided, iteration will stop *before* the first ancestor of the given kind.
            A typical use case is to pass `SymbolKind.File` or `SymbolKind.Package`.
        """
        parent = self.get_parent()
        if parent is not None:
            if up_to_symbol_kind is None or parent.symbol_kind != up_to_symbol_kind:
                yield parent
                yield from parent.iter_ancestors(up_to_symbol_kind=up_to_symbol_kind)

    def get_parent(self) -> Self | None:
        parent_root = self.symbol_root.get("parent")
        if parent_root is None:
            return None
        return self.__class__(parent_root)

    def find(
        self,
        name: str,
        substring_matching: bool = False,
        include_kinds: Sequence[SymbolKind] | None = None,
        exclude_kinds: Sequence[SymbolKind] | None = None,
    ) -> list[Self]:
        """
        Find all symbols within the symbol's subtree that match the given name.
        The name matching behavior depends on whether a qualified name or a simple name is provided.
        It is assumed that the provided name is a qualified name if it contains the `/` character.
        If substring matching is allowed, only the last element of the qualified name will be checked against
        the symbol name using substring matching.


        Examples:
        - Providing "foo" will find all symbols named "foo" regardless where they are contained in the symbol tree.
        - Providing "bar/foo" will only find symbols named "foo" that are direct children of a symbol called "bar".
        - Providing "foo/" will only find symbols named "foo" that are top-level symbols (have no parent).
        - Allowing substring matching with "bar" will find symbols with names containing "foo" anywhere in the symbol tree.
        - Allowing substring matching with "foo/" will find only top-level symbols with names containing "foo".
        - Allowing substring matching with "bar/foo" will find only symbols with names containing "foo" that are direct children of a symbol named "bar".

        :param name: the name of the symbol to find. Can use a qualified name (e.g. "class/method/inner_function")
            to restrict the search.
        :param substring_matching: whether to use substring matching for the symbol name.
            If a qualified name is provided, the last element of the qualified name will be checked against
            the symbol name using substring matching.
        :param include_kinds: an optional sequence of ints representing the LSP symbol kind.
            If provided, only symbols of the given kinds will be included in the result.
        :param exclude_kinds: If provided, symbols of the given kinds will be excluded from the result.

        """
        result = []

        def should_include(s: "Symbol") -> bool:
            if include_kinds is not None and s.symbol_kind not in include_kinds:
                return False
            if exclude_kinds is not None and s.symbol_kind in exclude_kinds:
                return False
            return Symbol.match_against_qualname(
                name_pattern=name,
                qual_name_parts=s.get_qualified_name_parts(),
                substring_matching=substring_matching,
            )

        def traverse(s: "Symbol") -> None:
            if should_include(s):
                result.append(s)
            for c in s.iter_children():
                traverse(c)

        traverse(self)
        return result

    def to_dict(
        self, kind: bool = False, location: bool = False, depth: int = 0, include_body: bool = False, include_children_body: bool = False
    ) -> dict[str, Any]:
        """
        Convert the symbol to a dictionary.

        :param kind: whether to include the kind of the symbol
        :param location: whether to include the location of the symbol
        :param depth: the depth of the symbol
        :param include_body: whether to include the body of the top-level symbol.
        :param include_children_body: whether to also include the body of the children.
            Note that the body of the children is part of the body of the parent symbol,
            so there is usually no need to set this to True unless you want process the output
            and pass the children without passing the parent body to the LM.
        :return: a dictionary representation of the symbol
        """
        result: dict[str, Any] = {"name": self.name}

        if kind:
            result["kind"] = self.kind

        if location:
            result["location"] = self.location.to_dict()

        if include_body:
            if self.body is None:
                log.warning("Requested body for symbol, but it is not present. The symbol might have been loaded with include_body=False.")
            result["body"] = self.body

        def add_children(s: Self) -> list[dict[str, Any]]:
            children = []
            for c in s.iter_children():
                children.append(
                    c.to_dict(
                        kind=kind,
                        location=location,
                        depth=depth - 1,
                        include_body=include_children_body,
                        include_children_body=include_children_body,
                    )
                )
            return children

        if depth > 0:
            result["children"] = add_children(self)

        return result


class SymbolManager:
    def __init__(self, lang_server: SyncLanguageServer, agent: "SerenaAgent") -> None:
        self.lang_server = lang_server
        self.agent = agent

    def _to_symbols(self, items: list[UnifiedSymbolInformation]) -> list[Symbol]:
        return [Symbol(s) for s in items]

    def find_by_name(
        self,
        name: str,
        within_relative_path: str | None = None,
        include_body: bool = False,
        include_kinds: Sequence[SymbolKind] | None = None,
        exclude_kinds: Sequence[SymbolKind] | None = None,
        substring_matching: bool = False,
    ) -> list[Symbol]:
        """
        Find all symbols that match the given name.

        :param name: the name of the symbol to find
        :param within_relative_path: pass a relative path to only consider symbols within this path.
            If a file is passed, only the symbols within this file will be considered.
            If a directory is passed, all files within this directory will be considered.
            If None, the entire codebase will be considered.
        :param include_body: whether to include the body of all symbols in the result.
            Note: you can filter out the bodies of the children if you set include_children_body=False
            in the to_dict method.
        :param include_kinds: an optional sequence of ints representing the LSP symbol kind.
            If provided, only symbols of the given kinds will be included in the result.
        :param exclude_kinds: If provided, symbols of the given kinds will be excluded from the result.
            Takes precedence over include_kinds.
        :param substring_matching: whether to use substring matching for the symbol name.
            If True, the symbol name will be matched if it contains the given name as a substring.
        :return: a list of symbols that match the given name
        """
        symbols: list[Symbol] = []
        symbol_roots = self.lang_server.request_full_symbol_tree(within_relative_path=within_relative_path, include_body=include_body)
        for root in symbol_roots:
            symbols.extend(
                Symbol(root).find(name, include_kinds=include_kinds, exclude_kinds=exclude_kinds, substring_matching=substring_matching)
            )
        return symbols

    def get_document_symbols(self, relative_path: str) -> list[Symbol]:
        symbol_dicts, roots = self.lang_server.request_document_symbols(relative_path, include_body=False)
        symbols = [Symbol(s) for s in symbol_dicts]
        return symbols

    def find_by_location(self, location: SymbolLocation) -> Symbol | None:
        if location.relative_path is None:
            return None
        symbol_dicts, roots = self.lang_server.request_document_symbols(location.relative_path, include_body=False)
        for symbol_dict in symbol_dicts:
            symbol = Symbol(symbol_dict)
            if symbol.location == location:
                return symbol
        return None

    def find_referencing_symbols(
        self,
        symbol_location: SymbolLocation,
        include_body: bool = False,
        include_kinds: Sequence[SymbolKind] | None = None,
        exclude_kinds: Sequence[SymbolKind] | None = None,
    ) -> list[Symbol]:
        """
        Find all symbols that reference the symbol at the given location.

        :param symbol_location: the location of the symbol for which to find references
        :param include_body: whether to include the body of all symbols in the result.
            Note: you can filter out the bodies of the children if you set include_children_body=False
            in the to_dict method.
        :param include_kinds: an optional sequence of ints representing the LSP symbol kind.
            If provided, only symbols of the given kinds will be included in the result.
        :param exclude_kinds: If provided, symbols of the given kinds will be excluded from the result.
            Takes precedence over include_kinds.
        :return: a list of symbols that reference the given symbol
        """
        if not symbol_location.has_position_in_file():
            raise ValueError("Symbol location does not contain a valid position in a file")
        assert symbol_location.relative_path is not None
        assert symbol_location.line is not None
        assert symbol_location.column is not None
        symbol_dicts = self.lang_server.request_referencing_symbols(
            relative_file_path=symbol_location.relative_path,
            line=symbol_location.line,
            column=symbol_location.column,
            include_imports=False,
            include_self=False,
            include_body=include_body,
        )

        if include_kinds is not None:
            symbol_dicts = [s for s in symbol_dicts if s["kind"] in include_kinds]

        if exclude_kinds is not None:
            symbol_dicts = [s for s in symbol_dicts if s["kind"] not in exclude_kinds]

        return self._to_symbols(symbol_dicts)

    @contextmanager
    def _edited_file(self, relative_path: str) -> Iterator[None]:
        with self.lang_server.open_file(relative_path) as file_buffer:
            yield
            root_path = self.lang_server.language_server.repository_root_path
            abs_path = os.path.join(root_path, relative_path)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(file_buffer.contents)
            self.agent.mark_file_modified(relative_path)

    @contextmanager
    def _edited_symbol_location(self, location: SymbolLocation) -> Iterator[Symbol]:
        symbol = self.find_by_location(location)
        if symbol is None:
            raise ValueError("Symbol not found/has no defined location within a file")
        assert location.relative_path is not None
        with self._edited_file(location.relative_path):
            yield symbol

    def replace_body(self, location: SymbolLocation, body: str) -> None:
        """
        Replace the body of the symbol at the given location with the given body

        :param location: the location of the symbol to replace
        :param body: the new body
        """
        # make sure body always ends with at least one newline
        if not body.endswith("\n"):
            body += "\n"
        with self._edited_symbol_location(location) as symbol:
            assert location.relative_path is not None
            start_pos = symbol.body_start_position
            end_pos = symbol.body_end_position
            if start_pos is None or end_pos is None:
                raise ValueError(f"Symbol at {location} does not have a defined body range.")
            # At this point, start_pos and end_pos are guaranteed to be Position objects
            self.lang_server.delete_text_between_positions(location.relative_path, start_pos, end_pos)
            self.lang_server.insert_text_at_position(location.relative_path, start_pos["line"], start_pos["character"], body)

    def insert_after(self, location: SymbolLocation, body: str) -> None:
        """
        Appends content after the given symbol

        :param location: the location of the symbol after which to add new lines
        :param body: the body of the entity to append
        """
        # make sure body always ends with at least one newline
        if not body.endswith("\n"):
            body += "\n"
        with self._edited_symbol_location(location) as symbol:
            pos = symbol.body_end_position
            if pos is None:
                raise ValueError(f"Symbol at {location} does not have a defined end position.")
            # At this point, pos is guaranteed to be a Position object
            assert location.relative_path is not None
            self.lang_server.insert_text_at_position(location.relative_path, pos["line"], pos["character"], body)

    def insert_before(self, location: SymbolLocation, body: str) -> None:
        """
        Inserts content before the given symbol

        :param location: the location of the symbol before which to add new lines
        :param body: the body of the entity to insert
        """
        # make sure body always ends with at least one newline
        if not body.endswith("\n"):
            body += "\n"
        with self._edited_symbol_location(location) as symbol:
            original_start_pos = symbol.body_start_position
            if original_start_pos is None:
                raise ValueError(f"Symbol at {location} does not have a defined start position.")
            # At this point, original_start_pos is guaranteed to be a Position object, so copying is safe.
            pos = copy(original_start_pos)
            assert location.relative_path is not None
            self.lang_server.insert_text_at_position(location.relative_path, pos["line"], pos["character"], body)

    def insert_at_line(self, relative_path: str, line: int, content: str) -> None:
        """
        Inserts content at the given line in the given file.

        :param line: the 0-based index of the line to insert content at
        :param content: the content to insert
        """
        with self._edited_file(relative_path):
            self.lang_server.insert_text_at_position(relative_path, line, 0, content)

    def delete_lines(self, relative_path: str, start_line: int, end_line: int) -> None:
        """
        Deletes lines in the given file.

        :param start_line: the 0-based index of the first line to delete (inclusive)
        :param end_line: the 0-based index of the last line to delete (inclusive)
        """
        with self._edited_file(relative_path):
            start_pos = Position(line=start_line, character=0)
            end_pos = Position(line=end_line + 1, character=0)
            self.lang_server.delete_text_between_positions(relative_path, start_pos, end_pos)
