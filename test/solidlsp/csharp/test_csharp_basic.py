import os

import pytest

from solidlsp import SolidLanguageServer
from solidlsp.ls_config import Language
from solidlsp.ls_utils import SymbolUtils


@pytest.mark.csharp
class TestCSharpLanguageServer:
    @pytest.mark.parametrize("language_server", [Language.CSHARP], indirect=True)
    def test_find_symbol(self, language_server: SolidLanguageServer) -> None:
        """Test finding symbols in the full symbol tree."""
        symbols = language_server.request_full_symbol_tree()
        assert SymbolUtils.symbol_tree_contains_name(symbols, "Program"), "Program class not found in symbol tree"
        assert SymbolUtils.symbol_tree_contains_name(symbols, "Calculator"), "Calculator class not found in symbol tree"
        assert SymbolUtils.symbol_tree_contains_name(symbols, "Add"), "Add method not found in symbol tree"

    @pytest.mark.parametrize("language_server", [Language.CSHARP], indirect=True)
    def test_get_document_symbols(self, language_server: SolidLanguageServer) -> None:
        """Test getting document symbols from a C# file."""
        file_path = os.path.join("Program.cs")
        symbols = language_server.request_document_symbols(file_path)

        # Check that we have symbols
        assert len(symbols) > 0

        # Flatten the symbols if they're nested
        if isinstance(symbols[0], list):
            symbols = symbols[0]

        # Look for expected classes
        class_names = [s.get("name") for s in symbols if s.get("kind") == 5]  # 5 is class
        assert "Program" in class_names
        assert "Calculator" in class_names

    @pytest.mark.parametrize("language_server", [Language.CSHARP], indirect=True)
    def test_find_referencing_symbols(self, language_server: SolidLanguageServer) -> None:
        """Test finding references using symbol selection range."""
        file_path = os.path.join("Program.cs")
        symbols = language_server.request_document_symbols(file_path)
        add_symbol = None
        # Handle nested symbol structure
        symbol_list = symbols[0] if symbols and isinstance(symbols[0], list) else symbols
        for sym in symbol_list:
            if sym.get("name") == "Add":
                add_symbol = sym
                break
        assert add_symbol is not None, "Could not find 'Add' method symbol in Program.cs"
        sel_start = add_symbol["selectionRange"]["start"]
        refs = language_server.request_references(file_path, sel_start["line"], sel_start["character"])
        assert any(
            "Program.cs" in ref.get("relativePath", "") for ref in refs
        ), "Program.cs should reference Add method (tried all positions in selectionRange)"

    @pytest.mark.parametrize("language_server", [Language.CSHARP], indirect=True)
    def test_nested_namespace_symbols(self, language_server: SolidLanguageServer) -> None:
        """Test getting symbols from nested namespace."""
        file_path = os.path.join("Models", "Person.cs")
        symbols = language_server.request_document_symbols(file_path)

        # Check that we have symbols
        assert len(symbols) > 0

        # Flatten the symbols if they're nested
        if isinstance(symbols[0], list):
            symbols = symbols[0]

        # Check that we have the Person class
        assert any(s.get("name") == "Person" and s.get("kind") == 5 for s in symbols)

        # Check for properties and methods
        symbol_names = [s.get("name") for s in symbols]
        assert "Name" in symbol_names
        assert "Age" in symbol_names
        assert "Email" in symbol_names
        assert "ToString" in symbol_names
        assert "IsAdult" in symbol_names
