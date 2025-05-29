import os

import pytest

from multilspy import SyncLanguageServer
from multilspy.multilspy_config import Language
from src.serena.symbol import SymbolManager

# Python test file path
PYTHON_TEST_REL_FILE_PATH = os.path.join("test_repo", "variables.py")

# TypeScript test file path
TYPESCRIPT_TEST_FILE = "index.ts"

# Expected deleted lines for Python VariableContainer
EXPECTED_DELETED_VARIABLE_CONTAINER_PYTHON = '''class VariableContainer:
    """Class that contains various variables."""

    # Class-level variables
    class_var = "Initial class value"

    reassignable_class_var = True
    reassignable_class_var = False  # Reassigned #noqa: PIE794

    # Class-level variable with type annotation
    typed_class_var: str = "typed value"

    def __init__(self):
        # Instance variables
        self.instance_var = "Initial instance value"
        self.reassignable_instance_var = 100

        # Instance variable with type annotation
        self.typed_instance_var: list[str] = ["item1", "item2"]

    def modify_instance_var(self):
        # Reassign instance variable
        self.instance_var = "Modified instance value"
        self.reassignable_instance_var = 200  # Reassigned

    def use_module_var(self):
        # Use module-level variables
        result = module_var + " used in method"
        other_result = reassignable_module_var + 5
        return result, other_result

    def use_class_var(self):
        # Use class-level variables
        result = VariableContainer.class_var + " used in method"
        other_result = VariableContainer.reassignable_class_var
        return result, other_result
'''

# Expected deleted lines for TypeScript DemoClass
EXPECTED_DELETED_DEMOCLASS_TYPESCRIPT = """export class DemoClass {
    value: number;
    constructor(value: number) {
        this.value = value;
    }
    printValue() {
        console.log(this.value);
    }
}"""


@pytest.mark.parametrize(
    "language_server, relative_file_path, symbol_name, expected_deleted_lines",
    [
        (
            Language.PYTHON,
            PYTHON_TEST_REL_FILE_PATH,
            "VariableContainer",
            EXPECTED_DELETED_VARIABLE_CONTAINER_PYTHON.strip().splitlines(),
        ),
        (
            Language.TYPESCRIPT,
            TYPESCRIPT_TEST_FILE,
            "DemoClass",
            EXPECTED_DELETED_DEMOCLASS_TYPESCRIPT.strip().splitlines(),
        ),
    ],
    indirect=["language_server"],
)
def test_delete_symbol_dry_run(
    language_server: SyncLanguageServer,
    relative_file_path: str,
    symbol_name: str,
    expected_deleted_lines: list[str],
):
    symbol_manager = SymbolManager(lang_server=language_server)
    code_diff = symbol_manager.delete_symbol(symbol_name, relative_file_path, dry_run=True)

    assert code_diff is not None
    assert code_diff.relative_path == relative_file_path
    assert len(code_diff.added_lines) == 0

    actual_deleted_lines = [line.strip() for _, line in code_diff.deleted_lines if line.strip()]

    # Normalize expected lines by stripping whitespace and removing empty lines
    normalized_expected_deleted_lines = [line.strip() for line in expected_deleted_lines if line.strip()]
    assert actual_deleted_lines == normalized_expected_deleted_lines
    assert code_diff.original_content != code_diff.modified_content


NEW_PYTHON_FUNCTION = """def new_inserted_function():
    print("This is a new function inserted before another.")"""

NEW_TYPESCRIPT_FUNCTION = """function newInsertedFunction(): void {
    console.log("This is a new function inserted before another.");
}"""


@pytest.mark.parametrize(
    "language_server, relative_file_path, symbol_name, new_content, expected_added_lines_content",
    [
        (
            Language.PYTHON,
            PYTHON_TEST_REL_FILE_PATH,
            "use_module_variables",
            NEW_PYTHON_FUNCTION,
            NEW_PYTHON_FUNCTION.strip().splitlines(),
        ),
        (
            Language.TYPESCRIPT,
            TYPESCRIPT_TEST_FILE,
            "helperFunction",
            NEW_TYPESCRIPT_FUNCTION,
            NEW_TYPESCRIPT_FUNCTION.strip().splitlines(),
        ),
    ],
    indirect=["language_server"],
)
def test_insert_before_symbol_dry_run(
    language_server: SyncLanguageServer,
    relative_file_path: str,
    symbol_name: str,
    new_content: str,
    expected_added_lines_content: list[str],
):
    symbol_manager = SymbolManager(lang_server=language_server)
    code_diff = symbol_manager.insert_before_symbol(symbol_name, relative_file_path, new_content, dry_run=True)

    assert code_diff is not None
    assert code_diff.relative_path == relative_file_path
    assert len(code_diff.deleted_lines) == 0
    actual_added_lines_content = [line.strip() for _, line in code_diff.added_lines if line.strip()]
    normalized_expected_added_lines = [line.strip() for line in expected_added_lines_content if line.strip()]

    assert actual_added_lines_content == normalized_expected_added_lines
    assert code_diff.original_content != code_diff.modified_content


NEW_PYTHON_VARIABLE = 'new_module_var = "Inserted after typed_module_var"'

NEW_TYPESCRIPT_FUNCTION_AFTER = """function newFunctionAfterClass(): void {
    console.log("This function is after DemoClass.");
}"""


@pytest.mark.parametrize(
    "language_server, relative_file_path, symbol_name, new_content, expected_added_lines_content",
    [
        (
            Language.PYTHON,
            PYTHON_TEST_REL_FILE_PATH,
            "typed_module_var",
            NEW_PYTHON_VARIABLE,
            [NEW_PYTHON_VARIABLE],
        ),
        (
            Language.TYPESCRIPT,
            TYPESCRIPT_TEST_FILE,
            "DemoClass",
            NEW_TYPESCRIPT_FUNCTION_AFTER,
            NEW_TYPESCRIPT_FUNCTION_AFTER.strip().splitlines(),
        ),
    ],
    indirect=["language_server"],
)
def test_insert_after_symbol_dry_run(
    language_server: SyncLanguageServer,
    relative_file_path: str,
    symbol_name: str,
    new_content: str,
    expected_added_lines_content: list[str],
):
    symbol_manager = SymbolManager(lang_server=language_server)
    code_diff = symbol_manager.insert_after_symbol(symbol_name, relative_file_path, new_content, dry_run=True)

    assert code_diff is not None
    assert code_diff.relative_path == relative_file_path
    assert len(code_diff.deleted_lines) == 0
    actual_added_lines_content = [line.strip() for _, line in code_diff.added_lines if line.strip()]
    normalized_expected_added_lines = [line.strip() for line in expected_added_lines_content if line.strip()]
    assert actual_added_lines_content == normalized_expected_added_lines
    assert code_diff.original_content != code_diff.modified_content


PYTHON_REPLACED_BODY = """        # This body has been replaced
        self.instance_var = "Replaced!"
        self.reassignable_instance_var = 999
"""

TYPESCRIPT_REPLACED_BODY = """        // This body has been replaced
        console.warn("New value: " + this.value);
"""

EXPECTED_ORIGINAL_MODIFY_INSTANCE_VAR_PYTHON = """    def modify_instance_var(self):
        # Reassign instance variable
        self.instance_var = "Modified instance value"
        self.reassignable_instance_var = 200  # Reassigned"""

# For single line original content, direct list is fine
EXPECTED_ORIGINAL_PRINTVALUE_TYPESCRIPT = ["    printValue() {", "        console.log(this.value);", "    }"]


@pytest.mark.parametrize(
    "language_server, relative_file_path, symbol_name, new_body, expected_original_lines_content, expected_modified_lines_content",
    [
        (
            Language.PYTHON,
            PYTHON_TEST_REL_FILE_PATH,
            "VariableContainer/modify_instance_var",
            PYTHON_REPLACED_BODY,
            EXPECTED_ORIGINAL_MODIFY_INSTANCE_VAR_PYTHON.strip().splitlines(),
            PYTHON_REPLACED_BODY.strip().splitlines(),
        ),
        (
            Language.TYPESCRIPT,
            TYPESCRIPT_TEST_FILE,
            "DemoClass/printValue",
            TYPESCRIPT_REPLACED_BODY,
            EXPECTED_ORIGINAL_PRINTVALUE_TYPESCRIPT,  # Already a list of lines
            TYPESCRIPT_REPLACED_BODY.strip().splitlines(),
        ),
    ],
    indirect=["language_server"],
)
def test_replace_body_dry_run(
    language_server: SyncLanguageServer,
    relative_file_path: str,
    symbol_name: str,
    new_body: str,
    expected_original_lines_content: list[str],
    expected_modified_lines_content: list[str],
):
    symbol_manager = SymbolManager(lang_server=language_server)
    code_diff = symbol_manager.replace_body(symbol_name, relative_file_path, new_body, dry_run=True)

    assert code_diff is not None
    assert code_diff.relative_path == relative_file_path

    actual_original_lines = [line.strip() for _, line in code_diff.deleted_lines if line.strip()]
    normalized_expected_original_lines = [line.strip() for line in expected_original_lines_content if line.strip()]
    assert actual_original_lines == normalized_expected_original_lines

    actual_modified_lines = [line.strip() for _, line in code_diff.added_lines if line.strip()]
    normalized_expected_modified_lines = [line.strip() for line in expected_modified_lines_content if line.strip()]
    assert actual_modified_lines == normalized_expected_modified_lines

    assert code_diff.original_content != code_diff.modified_content
