from collections.abc import Iterator
from typing import Any, Self

from sensai.util.string import ToStringMixin

from multilspy import SyncLanguageServer
from multilspy.multilspy_types import SymbolKind, UnifiedSymbolInformation


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
        return SymbolKind(self.s["kind"]).name

    @property
    def relative_path(self) -> str:
        return self.s["location"]["relativePath"]

    @property
    def line(self) -> int:
        return self.s["selectionRange"]["start"]["line"]

    @property
    def column(self) -> int:
        return self.s["selectionRange"]["start"]["character"]

    def iter_children(self) -> Iterator[Self]:
        for c in self.s["children"]:
            yield Symbol(c)

    def find(self, name: str) -> list[Self]:
        result = []

        def traverse(s: Self) -> None:
            if s.name == name:
                result.append(s)
            for c in s.iter_children():
                traverse(c)

        traverse(self)
        return result

    def to_dict(self, kind: bool = False, location: bool = False, depth: int = 0) -> dict[str, Any]:
        result: dict[str, Any] = {"name": self.name}

        if kind:
            result["kind"] = self.kind

        if location:
            result["location"] = {"relativePath": self.relative_path, "line": self.line, "column": self.column}

        def add_children(s: Self) -> list[dict[str, Any]]:
            children = []
            for c in s.iter_children():
                children.append(c.to_dict(kind=kind, location=location, depth=depth - 1))
            return children

        if depth > 0:
            result["children"] = add_children(self)

        return result


class SymbolRetriever:
    def __init__(self, lang_server: SyncLanguageServer) -> None:
        self.lang_server = lang_server

    def _to_symbols(self, items: list[UnifiedSymbolInformation]) -> list[Symbol]:
        return [Symbol(s) for s in items]

    def find(self, name: str) -> list[Symbol]:
        symbols: list[Symbol] = []
        symbol_roots = self.lang_server.request_full_symbol_tree()
        for root in symbol_roots:
            symbols.extend(Symbol(root).find(name))
        return symbols

    def find_references(self, relative_path: str, line: int, column: int) -> list[Symbol]:
        symbol_dicts = self.lang_server.request_referencing_symbols(
            relative_file_path=relative_path, line=line, column=column, include_imports=False, include_self=False, include_body=False
        )
        return self._to_symbols(symbol_dicts)
