import re

import pytest

from serena.text_utils import LineType, search_text


class TestTextUtils:
    def test_search_text_with_string_pattern(self):
        """Test searching with a simple string pattern."""
        content = """
        def hello_world():
            print("Hello, World!")
            return 42
        """

        # Search for a simple string pattern
        matches = search_text("print", content=content)

        assert len(matches) == 1
        assert matches[0].num_matched_lines == 1
        assert matches[0].start_line == 3
        assert matches[0].end_line == 3
        assert matches[0].lines[0].line_content.strip() == 'print("Hello, World!")'

    def test_search_text_with_regex_pattern(self):
        """Test searching with a regex pattern."""
        content = """
        class DataProcessor:
            def __init__(self, data):
                self.data = data

            def process(self):
                return [x * 2 for x in self.data if x > 0]

            def filter(self, predicate):
                return [x for x in self.data if predicate(x)]
        """

        # Search for a regex pattern matching method definitions
        pattern = r"def\s+\w+\s*\([^)]*\):"
        matches = search_text(pattern, content=content)

        assert len(matches) == 3
        assert matches[0].lines[0].match_type == LineType.MATCH
        assert "def __init__" in matches[0].lines[0].line_content
        assert "def process" in matches[1].lines[0].line_content
        assert "def filter" in matches[2].lines[0].line_content

    def test_search_text_with_compiled_regex(self):
        """Test searching with a pre-compiled regex pattern."""
        content = """
        import os
        import sys
        from pathlib import Path

        # Configuration variables
        DEBUG = True
        MAX_RETRIES = 3

        def configure_logging():
            log_level = "DEBUG" if DEBUG else "INFO"
            print(f"Setting log level to {log_level}")
        """

        # Search for variable assignments with a compiled regex
        pattern = re.compile(r"^\s*[A-Z_]+ = .+$")
        matches = search_text(pattern, content=content)

        assert len(matches) == 2
        assert "DEBUG = True" in matches[0].lines[0].line_content
        assert "MAX_RETRIES = 3" in matches[1].lines[0].line_content

    def test_search_text_with_context_lines(self):
        """Test searching with context lines before and after the match."""
        content = """
        def complex_function(a, b, c):
            # This is a complex function that does something.
            if a > b:
                return a * c
            elif b > a:
                return b * c
            else:
                return (a + b) * c
        """

        # Search with context lines
        matches = search_text("return", content=content, context_lines_before=1, context_lines_after=1)

        assert len(matches) == 3

        # Check the first match with context
        first_match = matches[0]
        assert len(first_match.lines) == 3
        assert first_match.lines[0].match_type == LineType.BEFORE_MATCH
        assert first_match.lines[1].match_type == LineType.MATCH
        assert first_match.lines[2].match_type == LineType.AFTER_MATCH

        # Verify the content of lines
        assert "if a > b:" in first_match.lines[0].line_content
        assert "return a * c" in first_match.lines[1].line_content
        assert "elif b > a:" in first_match.lines[2].line_content

    def test_search_text_with_multiline_match(self):
        """Test searching with multiline pattern matching."""
        content = """
        def factorial(n):
            if n <= 1:
                return 1
            else:
                return n * factorial(n-1)

        result = factorial(5)  # Should be 120
        """

        # Search for a pattern that spans multiple lines (if-else block)
        pattern = r"if.*?else.*?return"
        matches = search_text(pattern, content=content, allow_multiline_match=True)

        assert len(matches) == 1
        multiline_match = matches[0]
        assert multiline_match.num_matched_lines >= 3
        assert "if n <= 1:" in multiline_match.lines[0].line_content

        # All matched lines should have match_type == LineType.MATCH
        match_lines = [line for line in multiline_match.lines if line.match_type == LineType.MATCH]
        assert len(match_lines) >= 3

    def test_search_text_with_glob_pattern(self):
        """Test searching with glob-like patterns."""
        content = """
        class UserService:
            def get_user(self, user_id):
                return {"id": user_id, "name": "Test User"}

            def create_user(self, user_data):
                print(f"Creating user: {user_data}")
                return {"id": 123, **user_data}

            def update_user(self, user_id, user_data):
                print(f"Updating user {user_id} with {user_data}")
                return True
        """

        # Search with a glob pattern for all user methods
        matches = search_text("*_user*", content=content, is_glob=True)

        assert len(matches) == 3
        assert "get_user" in matches[0].lines[0].line_content
        assert "create_user" in matches[1].lines[0].line_content
        assert "update_user" in matches[2].lines[0].line_content

    def test_search_text_with_complex_glob_pattern(self):
        """Test searching with more complex glob patterns."""
        content = """
        def process_data(data):
            return [transform(item) for item in data]

        def transform(item):
            if isinstance(item, dict):
                return {k: v.upper() if isinstance(v, str) else v for k, v in item.items()}
            elif isinstance(item, list):
                return [x * 2 for x in item if isinstance(x, (int, float))]
            elif isinstance(item, str):
                return item.upper()
            else:
                return item
        """

        # Search with a simplified glob pattern to find all isinstance occurrences
        matches = search_text("*isinstance*", content=content, is_glob=True)

        # Should match lines with isinstance(item, dict) and isinstance(item, list)
        assert len(matches) >= 2
        instance_matches = [
            line.line_content
            for match in matches
            for line in match.lines
            if line.match_type == LineType.MATCH and "isinstance(item," in line.line_content
        ]
        assert len(instance_matches) >= 2
        assert any("isinstance(item, dict)" in line for line in instance_matches)
        assert any("isinstance(item, list)" in line for line in instance_matches)

    def test_search_text_no_matches(self):
        """Test searching with a pattern that doesn't match anything."""
        content = """
        def calculate_average(numbers):
            if not numbers:
                return 0
            return sum(numbers) / len(numbers)
        """

        # Search for a pattern that doesn't exist in the content
        matches = search_text("missing_function", content=content)

        assert len(matches) == 0

    def test_search_text_invalid_regex(self):
        """Test searching with an invalid regex pattern raises ValueError."""
        content = "def example(): pass"

        # Search with an invalid regex pattern (unmatched parenthesis)
        with pytest.raises(ValueError):
            search_text("example(", content=content)
