import logging
from collections.abc import Iterator, Sequence
from typing import Any, Self

from sensai.util.string import ToStringMixin

from multilspy import SyncLanguageServer
from multilspy.multilspy_types import SymbolKind, UnifiedSymbolInformation

log = logging.getLogger(__name__)


class Symbol(ToStringMixin):
    def __init__(self, s: UnifiedSymbolInformation) -> None:
        self.s = s

    def _tostring_includes(self) -> list[str]:
        return []

    def _tostring_additional_entries(self) -> dict[str, Any]:
        return dict(name=self.name, kind=self.kind, num_children=len(self.s["children"]))

    @property
    def name(self) -> str:
        return self.s["name"]

    @property
    def kind(self) -> str:
        return SymbolKind(self.symbol_kind).name

    @property
    def symbol_kind(self) -> SymbolKind:
        return self.s["kind"]

    @property
    def relative_path(self) -> str:
        return self.s["location"]["relativePath"]

    @property
    def line(self) -> int:
        return self.s["selectionRange"]["start"]["line"]

    @property
    def column(self) -> int:
        return self.s["selectionRange"]["start"]["character"]

    @property
    def body(self) -> str | None:
        return self.s.get("body")

    def iter_children(self) -> Iterator["Symbol"]:
        for c in self.s["children"]:
            yield Symbol(c)

    def find(
        self,
        name: str,
        substring_matching: bool = False,
        include_kinds: Sequence[SymbolKind] | None = None,
        exclude_kinds: Sequence[SymbolKind] | None = None,
    ) -> list["Symbol"]:
        result = []

        def should_include(s: "Symbol") -> bool:
            name_match = (substring_matching and name in s.name) or name == s.name
            kind_include_match = include_kinds is None or s.symbol_kind in include_kinds
            kind_exclude_match = exclude_kinds is None or s.symbol_kind not in exclude_kinds

            return name_match and kind_include_match and kind_exclude_match

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
            result["location"] = {"relativePath": self.relative_path, "line": self.line, "column": self.column}

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


class SymbolRetriever:
    def __init__(self, lang_server: SyncLanguageServer) -> None:
        self.lang_server = lang_server

    def _to_symbols(self, items: list[UnifiedSymbolInformation]) -> list[Symbol]:
        return [Symbol(s) for s in items]

    def find(
        self,
        name: str,
        dir_relative_path: str | None = None,
        include_body: bool = False,
        include_kinds: Sequence[SymbolKind] | None = None,
        exclude_kinds: Sequence[SymbolKind] | None = None,
        substring_matching: bool = False,
    ) -> list[Symbol]:
        """
        Find all symbols that match the given name.

        :param name: the name of the symbol to find
        :param dir_relative_path: pass a directory relative path to only consider symbols within this directory.
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
        symbol_roots = self.lang_server.request_full_symbol_tree(start_package_relative_path=dir_relative_path, include_body=include_body)
        for root in symbol_roots:
            symbols.extend(
                Symbol(root).find(name, include_kinds=include_kinds, exclude_kinds=exclude_kinds, substring_matching=substring_matching)
            )
        return symbols

    def find_references(
        self,
        relative_path: str,
        line: int,
        column: int,
        include_body: bool = False,
        include_kinds: Sequence[SymbolKind] | None = None,
        exclude_kinds: Sequence[SymbolKind] | None = None,
    ) -> list[Symbol]:
        """
        Find all symbols that reference the given symbol.

        :param relative_path: the relative path to the file containing the symbol
        :param line: the line number of the symbol (0-indexed).
        :param column: the column number of the symbol. Note that this usually corresponds to the
            column in `selectionRange` of the symbol (as opposed to the `range`).
        :param include_body: whether to include the body of all symbols in the result.
            Note: you can filter out the bodies of the children if you set include_children_body=False
            in the to_dict method.
        :param include_kinds: an optional sequence of ints representing the LSP symbol kind.
            If provided, only symbols of the given kinds will be included in the result.
        :param exclude_kinds: If provided, symbols of the given kinds will be excluded from the result.
            Takes precedence over include_kinds.
        :return: a list of symbols that reference the given symbol
        """
        symbol_dicts = self.lang_server.request_referencing_symbols(
            relative_file_path=relative_path, line=line, column=column, include_imports=False, include_self=False, include_body=include_body
        )

        if include_kinds is not None:
            symbol_dicts = [s for s in symbol_dicts if s["kind"] in include_kinds]

        if exclude_kinds is not None:
            symbol_dicts = [s for s in symbol_dicts if s["kind"] not in exclude_kinds]

        return self._to_symbols(symbol_dicts)
