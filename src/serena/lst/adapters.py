from __future__ import annotations

from abc import ABC, abstractmethod
from collections import abc
from dataclasses import dataclass
from typing import Any, ClassVar

from serena.util.class_decorators import singleton
from solidlsp.ls_config import Language


@dataclass(frozen=True)
class LosslessSymbolContext:
    """
    Minimal, language-agnostic context describing the symbol for which an LST shall be generated.
    """

    name: str
    name_path: str
    kind: str
    relative_path: str
    range: dict[str, Any]
    selection_range: dict[str, Any] | None
    language: Language


class LosslessSemanticTreeAdapter(ABC):
    """
    Base class for language-specific adapters that can produce OpenRewrite-style LSTs.
    """

    language: ClassVar[Language]

    @abstractmethod
    def build(
        self,
        context: LosslessSymbolContext,
        file_content: str,
        include_source_text: bool,
        max_depth: int,
    ) -> dict[str, Any]:
        """
        Produce an LST dictionary for the given symbol context.
        """


@singleton
class LosslessSemanticTreeAdapterRegistry:
    """
    Registry keeping track of available language-specific LST adapters.
    """

    def __init__(self) -> None:
        self._adapter_map: dict[Language, LosslessSemanticTreeAdapter] = {}
        self._initialize()

    def _initialize(self) -> None:
        from .cpp_adapter import CppLosslessSemanticTreeAdapter

        cpp_adapter = CppLosslessSemanticTreeAdapter()
        self._adapter_map[cpp_adapter.language] = cpp_adapter

    def get_adapter(self, language: Language) -> LosslessSemanticTreeAdapter | None:
        return self._adapter_map.get(language)

    def supported_languages(self) -> abc.KeysView[Language]:
        return self._adapter_map.keys()
