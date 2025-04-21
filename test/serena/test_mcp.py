"""Tests for the mcp.py module in serena."""

from serena.mcp import _process_parameter_descriptions


def test_basic_parameter_processing() -> None:
    """Test basic functionality with parameters and return description."""
    params_props = {
        "name": {},
        "age": {},
    }
    func_doc = """This is a test function.

    :param name: The person's name
    :param age: The person's age
    :return: A greeting message
    """

    result_doc = _process_parameter_descriptions(params_props, func_doc)

    assert "param" not in result_doc
    assert "Returns A greeting message" in result_doc
    assert result_doc == "This is a test function. Returns A greeting message."
    assert params_props["name"]["description"] == "The person's name."
    assert params_props["age"]["description"] == "The person's age."


def test_no_parameters() -> None:
    """Test with a docstring that has no parameters."""
    params_props = {}
    func_doc = """This is a test function with no parameters.

    :return: A simple result
    """

    result_doc = _process_parameter_descriptions(params_props, func_doc)

    assert result_doc == "This is a test function with no parameters. Returns A simple result."
    assert len(params_props) == 0


def test_no_return_description() -> None:
    """Test with a docstring that has no return description."""
    params_props = {"name": {}}
    func_doc = """This is a test function.

    :param name: The parameter name
    """

    result_doc = _process_parameter_descriptions(params_props, func_doc)

    assert result_doc == "This is a test function."
    assert "description" in params_props["name"]
    assert params_props["name"]["description"] == "The parameter name."


def test_existing_parameter_description() -> None:
    """Test when parameters already have descriptions in the properties dict."""
    params_props = {
        "name": {"description": "Existing name description"},
        "age": {},
    }
    func_doc = """This is a test function.

    :param name: The person's name
    :param age: The person's age
    """

    result_doc = _process_parameter_descriptions(params_props, func_doc)

    assert result_doc == "This is a test function."
    assert params_props["name"]["description"] == "Existing name description"
    assert params_props["age"]["description"] == "The person's age."


def test_parameter_not_in_docstring() -> None:
    """Test when a parameter in properties is not in the docstring."""
    params_props = {
        "name": {},
        "missing_param": {},
    }
    func_doc = """This is a test function.

    :param name: The person's name
    """

    result_doc = _process_parameter_descriptions(params_props, func_doc)

    assert result_doc == "This is a test function."
    assert params_props["name"]["description"] == "The person's name."
    assert "description" not in params_props["missing_param"]


def test_multiline_docstring() -> None:
    """Test with a complex multi-line docstring."""
    params_props = {
        "project_file_path": {},
        "host": {},
        "port": {},
    }
    func_doc = """Create an MCP server.

    This function creates and configures a Model Context Protocol server
    with the specified settings.

    :param project_file_path: The path to the project file, or None
    :param host: The host to bind to
    :param port: The port to bind to
    :return: A configured FastMCP server instance
    """

    result_doc = _process_parameter_descriptions(params_props, func_doc)

    assert "param" not in result_doc
    assert "Returns A configured FastMCP server instance" in result_doc
    assert params_props["project_file_path"]["description"] == "The path to the project file, or None."
    assert params_props["host"]["description"] == "The host to bind to."
    assert params_props["port"]["description"] == "The port to bind to."


def test_capitalization_and_periods() -> None:
    """Test that the function properly capitalizes descriptions and adds periods."""
    params_props = {
        "param1": {},
        "param2": {},
        "param3": {},
    }
    func_doc = """Test function.

    :param param1: lowercase description
    :param param2: description with period.
    :param param3: description with Capitalized word.
    """

    _ = _process_parameter_descriptions(params_props, func_doc)

    assert params_props["param1"]["description"] == "Lowercase description."
    assert params_props["param2"]["description"] == "Description with period."
    assert params_props["param3"]["description"] == "Description with Capitalized word."
