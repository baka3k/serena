"""Simple script to verify our fix for the make_tool function."""

import docstring_parser
from typing import Any, Optional


def make_tool_simplified(func_doc: str) -> str:
    """Simplified version of make_tool function to test our fix."""
    docstring = docstring_parser.parse(func_doc)

    # Mount the tool description as a combination of the docstring description and
    # the return value description, if it exists.
    if docstring.description:
        func_doc = f"{docstring.description.strip().strip('.')}."
    else:
        func_doc = ""
    if (docstring.returns) and (docstring_returns := docstring.returns.description):
        # Only add a space before "Returns" if func_doc is not empty
        prefix = " " if func_doc else ""
        func_doc = f"{func_doc}{prefix}Returns {docstring_returns.strip().strip('.')}."
    
    return func_doc


def test_with_description():
    """Test with a normal docstring that has a description."""
    doc = """This is a test function.

    :param name: The person's name
    :param age: The person's age
    :return: A greeting message
    """
    result = make_tool_simplified(doc)
    assert result == "This is a test function. Returns A greeting message."
    print("✅ Test with description passed")


def test_without_description():
    """Test with a docstring that has no description."""
    doc = """
    :param name: The person's name
    :param age: The person's age
    :return: A greeting message
    """
    result = make_tool_simplified(doc)
    assert result == "Returns A greeting message."
    print("✅ Test without description passed")


def test_empty_docstring():
    """Test with an empty docstring."""
    doc = ""
    result = make_tool_simplified(doc)
    assert result == ""
    print("✅ Test with empty docstring passed")


def test_no_return():
    """Test with a docstring that has no return description."""
    doc = """This is a test function.

    :param name: The person's name
    :param age: The person's age
    """
    result = make_tool_simplified(doc)
    assert result == "This is a test function."
    print("✅ Test with no return passed")


if __name__ == "__main__":
    test_with_description()
    test_without_description()
    test_empty_docstring()
    test_no_return()
    print("All tests passed!")