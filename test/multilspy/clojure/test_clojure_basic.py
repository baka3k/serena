import pytest

from multilspy.language_server import SyncLanguageServer
from multilspy.multilspy_config import Language
from multilspy.multilspy_types import Position

@pytest.mark.clojure
class TestLanguageServerBasics:

    @pytest.mark.parametrize("language_server", [Language.CLOJURE], indirect=True)
    def test_basic_definition(self, language_server: SyncLanguageServer):
        """
        Test finding definition of 'greet' function call in core.clj
        """

        filepath = "src/test_app/core.clj"
        result = language_server.request_definition(filepath, 20, 12)  # Position of 'greet' in (greet "World")

        assert isinstance(result, list)
        assert len(result) >= 1
        
        definition = result[0]
        assert definition["relativePath"] == "src/test_app/core.clj"
        assert definition["range"]["start"]["line"] == 2, \
            "Should find the definition of greet function at line 2"


    @pytest.mark.parametrize("language_server", [Language.CLOJURE], indirect=True)
    def test_cross_file_references(self, language_server: SyncLanguageServer):
        """
        Test finding references to 'multiply' function from core.clj
        """
        filepath = "src/test_app/core.clj"
        result = language_server.request_references(filepath, 12, 6)

        assert isinstance(result, list)
        assert len(result) >= 2  # Should find definition + usage in utils.clj

        usage_found = any(
            item["relativePath"] == "src/test_app/utils.clj" and 
            item["range"]["start"]["line"] == 6  # multiply usage in calculate-area
            for item in result
        )
        assert usage_found, "Should find multiply usage in utils.clj"


    @pytest.mark.parametrize("language_server", [Language.CLOJURE], indirect=True)
    def test_completions(self, language_server: SyncLanguageServer):
        filepath = "src/test_app/utils.clj"
        with language_server.open_file(filepath):
            # After "core/" in calculate-area
            result = language_server.request_completions(filepath, 6, 8)

            assert isinstance(result, list)
            assert len(result) > 0

            # Should find multiply and other core functions
            completion_texts = [item["completionText"] for item in result]
            assert any("multiply" in text for text in completion_texts)


    @pytest.mark.parametrize("language_server", [Language.CLOJURE], indirect=True)
    def test_document_symbols(self, language_server: SyncLanguageServer):
        filepath = "src/test_app/core.clj"
        symbols, _ = language_server.request_document_symbols(filepath)

        assert isinstance(symbols, list)
        assert len(symbols) >= 4  # greet, add, multiply, -main functions

        # Check that we find the expected function symbols
        symbol_names = [symbol["name"] for symbol in symbols]
        expected_functions = ["greet", "add", "multiply", "-main"]
        
        for func_name in expected_functions:
            assert func_name in symbol_names, f"Should find {func_name} function in symbols"


    @pytest.mark.parametrize("language_server", [Language.CLOJURE], indirect=True)
    def test_hover(self, language_server: SyncLanguageServer):
        # Test hover on greet function
        filepath = "src/test_app/core.clj" 
        result = language_server.request_hover(filepath, 2, 7)  # Position on 'greet' function name

        if result is not None:
            assert "contents" in result
            # Should contain function signature or documentation
            contents = result["contents"]
            if isinstance(contents, str):
                assert "greet" in contents.lower()
            elif isinstance(contents, dict) and "value" in contents:
                assert "greet" in contents["value"].lower()


    @pytest.mark.parametrize("language_server", [Language.CLOJURE], indirect=True)
    def test_workspace_symbols(self, language_server: SyncLanguageServer):
        # Search for functions containing "add"
        result = language_server.request_workspace_symbol("add")

        if result is not None:
            assert isinstance(result, list)
            
            # Should find the 'add' function
            symbol_names = [symbol["name"] for symbol in result]
            assert any("add" in name.lower() for name in symbol_names)

    @pytest.mark.parametrize("language_server", [Language.CLOJURE], indirect=True)
    def test_namespace_functions(self, language_server: SyncLanguageServer):
        """Test definition lookup for core/greet usage in utils.clj"""

        filepath = "src/test_app/utils.clj"
         # Position of 'greet' in core/greet call
        result = language_server.request_definition(filepath, 11, 25)

        assert isinstance(result, list)
        assert len(result) >= 1
        
        definition = result[0]
        assert definition["relativePath"] == "src/test_app/core.clj",\
            "Should find the definition of greet in core.clj"

