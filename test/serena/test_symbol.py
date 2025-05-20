import pytest

from src.serena.symbol import Symbol


class TestSymbolNameMatching:
    def _create_assertion_error_message(
        self,
        name_pattern: str,
        qual_name_parts: list[str],
        is_substring_match: bool,
        expected_result: bool,
        actual_result: bool,
    ) -> str:
        """Helper to create a detailed error message for assertions."""
        qnp_repr = "/".join(qual_name_parts)

        return (
            f"Pattern '{name_pattern}' (substring: {is_substring_match}) vs "
            f"Qualname parts {qual_name_parts} (as '{qnp_repr}'). "
            f"Expected: {expected_result}, Got: {actual_result}"
        )

    @pytest.mark.parametrize(
        "name_pattern, qual_name_parts, is_substring_match, expected",
        [
            # Exact matches (is_substring_match=False)
            pytest.param("foo", ["foo"], False, True, id="'foo' matches 'foo' exactly"),
            pytest.param("foo", ["bar", "foo"], False, True, id="'foo' matches 'bar,foo' exactly"),
            pytest.param("foo", ["foobar"], False, False, id="'foo' does not match 'foobar' exactly"),
            pytest.param("foo", ["bar", "foobar"], False, False, id="'foo' does not match 'bar,foobar' exactly"),
            pytest.param("foo", ["path", "to", "foo"], False, True, id="'foo' matches 'path,to,foo' exactly"),
            # Substring matches (is_substring_match=True)
            pytest.param("foo", ["foobar"], True, True, id="'foo' matches 'foobar' as substring"),
            pytest.param("foo", ["bar", "foobar"], True, True, id="'foo' matches 'bar,foobar' as substring"),
            pytest.param("foo", ["barfoo"], True, True, id="'foo' matches 'barfoo' as substring"),
            pytest.param("foo", ["baz"], True, False, id="'foo' does not match 'baz' as substring"),
            pytest.param("foo", ["bar", "baz"], True, False, id="'foo' does not match 'bar,baz' as substring"),
            pytest.param("foo", ["my_foobar_func"], True, True, id="'foo' matches 'my_foobar_func' as substring"),
            pytest.param("foo", ["ClassA", "my_foobar_method"], True, True, id="'foo' matches 'ClassA,my_foobar_method' as substring"),
            pytest.param("foo", ["my_bar_func"], True, False, id="'foo' does not match 'my_bar_func' as substring"),
        ],
    )
    def test_match_simple_name(self, name_pattern, qual_name_parts, is_substring_match, expected):
        """Tests matching for simple names (no '/' in pattern)."""
        result = Symbol.match_against_qualname(name_pattern, qual_name_parts, is_substring_match)
        error_msg = self._create_assertion_error_message(name_pattern, qual_name_parts, is_substring_match, expected, result)
        assert result == expected, error_msg

    @pytest.mark.parametrize(
        "name_pattern, qual_name_parts, is_substring_match, expected",
        [
            # Exact matches (is_substring_match=False)
            pytest.param("bar/foo", ["bar", "foo"], False, True, id="'bar/foo' matches 'bar,foo' exactly"),
            pytest.param(
                "bar/foo", ["bar", "foo", "baz"], False, False, id="'bar/foo' does not match 'bar,foo,baz' exactly (len mismatch)"
            ),
            pytest.param("bar/foo", ["bar"], False, False, id="'bar/foo' does not match 'bar' exactly (len mismatch)"),
            pytest.param("bar/foo", ["baz", "foo"], False, False, id="'bar/foo' does not match 'baz,foo' exactly (first mismatch)"),
            pytest.param("bar/foo", ["bar", "baz"], False, False, id="'bar/foo' does not match 'bar,baz' exactly (last mismatch)"),
            # from docstring examples
            pytest.param("bar/foo", ["foo"], False, False, id="'bar/foo' does not match 'foo' exactly"),
            pytest.param("bar/foo", ["other", "foo"], False, False, id="'bar/foo' does not match 'other,foo' exactly"),
            pytest.param("bar/foo", ["bar", "otherfoo"], False, False, id="'bar/foo' does not match 'bar,otherfoo' exactly"),
            # Substring matches (is_substring_match=True)
            pytest.param("bar/foo", ["bar", "foobar"], True, True, id="'bar/foo' matches 'bar,foobar' as substring"),
            pytest.param("bar/foo", ["bar", "bazfoo"], True, True, id="'bar/foo' matches 'bar,bazfoo' as substring"),
            pytest.param("bar/fo", ["bar", "foo"], True, True, id="'bar/fo' matches 'bar,foo' as substring"),
            pytest.param("bar/foo", ["bar", "baz"], True, False, id="'bar/foo' does not match 'bar,baz' as substring (last no substr)"),
            pytest.param(
                "bar/foo", ["baz", "foobar"], True, False, id="'bar/foo' does not match 'baz,foobar' as substring (first mismatch)"
            ),
            # from docstring examples
            pytest.param("bar/foo", ["bar", "my_foobar_method"], True, True, id="'bar/foo' matches 'bar,my_foobar_method' as substring"),
            pytest.param(
                "bar/foo", ["bar", "another_method"], True, False, id="'bar/foo' does not match 'bar,another_method' as substring"
            ),
            pytest.param(
                "bar/foo", ["other", "my_foobar_method"], True, False, id="'bar/foo' does not match 'other,my_foobar_method' as substring"
            ),
            pytest.param("bar/f", ["bar", "foo"], True, True, id="'bar/f' matches 'bar,foo' as substring"),
        ],
    )
    def test_match_name_pattern_ends_without_slash(self, name_pattern, qual_name_parts, is_substring_match, expected):
        """Tests matching for qualified names (e.g. 'module/class/func')."""
        result = Symbol.match_against_qualname(name_pattern, qual_name_parts, is_substring_match)
        error_msg = self._create_assertion_error_message(name_pattern, qual_name_parts, is_substring_match, expected, result)
        assert result == expected, error_msg

    @pytest.mark.parametrize(
        "name_pattern, qual_name_parts, is_substring_match, expected",
        [
            # Exact matches (is_substring_match=False)
            pytest.param("foo/", ["foo"], False, True, id="'foo/' matches 'foo' exactly"),
            pytest.param("bar/foo/", ["bar", "foo"], False, True, id="'bar/foo/' matches 'bar,foo' exactly"),
            pytest.param("foo/", ["foobar"], False, False, id="'foo/' does not match 'foobar' exactly"),
            pytest.param("foo/", ["bar", "foo"], False, False, id="'foo/' does not match 'bar,foo' exactly (not toplevel)"),
            # Substring matches (is_substring_match=True)
            pytest.param("foo/", ["foobar"], True, True, id="'foo/' matches 'foobar' as substring"),
            pytest.param(
                "bar/foo/", ["bar", "the_foobar_method"], True, True, id="'bar/foo/' matches 'bar,the_foobar_method' as substring"
            ),
            pytest.param(
                "bar/foo/",
                ["bar", "another_method"],
                True,
                False,
                id="'bar/foo/' does not match 'bar,another_method' as substring (last no substr)",
            ),
            pytest.param(
                "bar/foo/",
                ["baz", "the_foobar_method"],
                True,
                False,
                id="'bar/foo/' does not match 'baz,the_foobar_method' as substring (first mismatch)",
            ),
            # from docstring examples
            pytest.param("foo/", ["my_foobar_func"], True, True, id="'foo/' matches 'my_foobar_func' as substring"),
            pytest.param(
                "foo/",
                ["ClassA", "my_foobar_method"],
                True,
                False,
                id="'foo/' does not match 'ClassA,my_foobar_method' as substring (not toplevel)",
            ),
            pytest.param("foo/", ["foo"], True, True, id="'foo/' matches 'foo' as substring (exact is substr)"),
        ],
    )
    def test_match_name_pattern_ending_in_slash(self, name_pattern, qual_name_parts, is_substring_match, expected):
        """Tests matching for patterns ending with '/' (prefix/namespace style)."""
        result = Symbol.match_against_qualname(name_pattern, qual_name_parts, is_substring_match)
        error_msg = self._create_assertion_error_message(name_pattern, qual_name_parts, is_substring_match, expected, result)
        assert result == expected, error_msg
