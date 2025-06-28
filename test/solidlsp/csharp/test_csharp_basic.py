"""
Basic tests for C# language server functionality.
"""

import os
from pathlib import Path

import pytest

from solidlsp import SolidLanguageServer
from solidlsp.ls_config import Language
from test.conftest import create_ls


@pytest.fixture
def csharp_ls() -> SolidLanguageServer:
    """Create a C# language server instance for testing."""
    repo_path = Path(__file__).parent.parent.parent / "resources" / "repos" / "csharp" / "test_repo"
    server = create_ls(Language.CSHARP, str(repo_path))
    server.start()
    try:
        yield server
    finally:
        server.stop()


class TestCSharpBasic:
    """Basic tests for C# language server."""

    def test_language_server_starts(self, csharp_ls: SolidLanguageServer):
        """Test that the C# language server starts successfully."""
        assert csharp_ls is not None
        assert csharp_ls.language == Language.CSHARP

    def test_get_document_symbols(self, csharp_ls: SolidLanguageServer):
        """Test getting document symbols from a C# file."""
        file_path = os.path.join("Program.cs")
        symbols = csharp_ls.request_document_symbols(file_path)

        # Check that we have symbols
        assert len(symbols) > 0

        # Flatten the symbols if they're nested
        if isinstance(symbols[0], list):
            symbols = symbols[0]

        # Look for expected classes
        class_names = [s.get("name") for s in symbols if s.get("kind") == 5]  # 5 is class
        assert "Program" in class_names
        assert "Calculator" in class_names

    def test_find_definition(self, csharp_ls: SolidLanguageServer):
        """Test finding definition of a symbol."""
        file_path = os.path.join("Program.cs")

        # Open the file first
        with csharp_ls.open_file(file_path):
            # Find usage of Calculator class (line 11, column 28)
            definitions = csharp_ls.request_definition(file_path, 11, 28)

            # Should find the Calculator class definition
            assert len(definitions) > 0
            assert any("Calculator" in str(d) for d in definitions)

    def test_find_references(self, csharp_ls: SolidLanguageServer):
        """Test finding references to a symbol."""
        file_path = os.path.join("Program.cs")

        # Open the file first
        with csharp_ls.open_file(file_path):
            # Find references to the Add method (line 19)
            references = csharp_ls.request_references(file_path, 19, 20, include_declaration=True)

            # Should find at least the definition and one usage
            assert len(references) >= 2

    def test_nested_namespace_symbols(self, csharp_ls: SolidLanguageServer):
        """Test getting symbols from nested namespace."""
        file_path = os.path.join("Models", "Person.cs")
        symbols = csharp_ls.request_document_symbols(file_path)

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
