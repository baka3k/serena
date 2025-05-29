import os
import shutil
import tempfile
from abc import abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from multilspy.multilspy_config import Language
from serena.symbol import CodeDiff
from src.serena.symbol import SymbolManager
from test.conftest import create_ls, get_repo_path


class EditingTest:
    def __init__(self, language: Language, rel_path: str):
        """
        :param language: the language
        :param rel_path: the relative path of the edited file
        """
        self.rel_path = rel_path
        self.language = language
        self.original_repo_path = get_repo_path(language)
        self.repo_path: Path | None = None

    @contextmanager
    def _setup(self) -> Iterator[SymbolManager]:
        """Context manager for setup/teardown with a temporary directory, providing the symbol manager."""
        temp_dir = Path(tempfile.mkdtemp())
        self.repo_path = temp_dir / self.original_repo_path.name
        try:
            shutil.copytree(self.original_repo_path, self.repo_path)
            language_server = create_ls(self.language, str(self.repo_path))
            language_server.start()
            yield SymbolManager(lang_server=language_server)
        finally:
            shutil.rmtree(temp_dir)

    def _read_file(self, rel_path: str) -> str:
        """Read the content of a file in the test repository."""
        file_path = self.repo_path / rel_path
        with open(file_path, encoding="utf-8") as f:
            return f.read()

    def run_test(self) -> None:
        with self._setup() as symbol_manager:
            content_before = self._read_file(self.rel_path)
            self._apply_edit(symbol_manager)
            content_after = self._read_file(self.rel_path)
            code_diff = CodeDiff(self.rel_path, original_content=content_before, modified_content=content_after)
            self._test_diff(code_diff)

    @abstractmethod
    def _apply_edit(self, symbol_manager: SymbolManager) -> None:
        pass

    @abstractmethod
    def _test_diff(self, code_diff: CodeDiff) -> None:
        pass


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


class DeleteSymbolTest(EditingTest):
    def __init__(self, language: Language, rel_path: str, deleted_symbol: str, expected_deleted_lines: str):
        super().__init__(language, rel_path)
        self.expected_deleted_lines = expected_deleted_lines
        self.deleted_symbol = deleted_symbol
        self.rel_path = rel_path

    def _apply_edit(self, symbol_manager: SymbolManager) -> None:
        symbol_manager.delete_symbol(self.deleted_symbol, self.rel_path)

    def _test_diff(self, code_diff: CodeDiff) -> None:
        assert code_diff.original_content != code_diff.modified_content
        actual_deleted_lines = [line.strip() for _, line in code_diff.deleted_lines if line.strip()]
        normalized_expected_deleted_lines = [line.strip() for line in self.expected_deleted_lines.splitlines() if line.strip()]
        assert actual_deleted_lines == normalized_expected_deleted_lines


@pytest.mark.parametrize(
    "test_case",
    [
        DeleteSymbolTest(
            Language.PYTHON,
            PYTHON_TEST_REL_FILE_PATH,
            "VariableContainer",
            EXPECTED_DELETED_VARIABLE_CONTAINER_PYTHON,
        ),
        DeleteSymbolTest(
            Language.TYPESCRIPT,
            TYPESCRIPT_TEST_FILE,
            "DemoClass",
            EXPECTED_DELETED_DEMOCLASS_TYPESCRIPT,
        ),
    ],
)
def test_delete_symbol(test_case):
    test_case.run_test()


NEW_PYTHON_FUNCTION = """def new_inserted_function():
    print("This is a new function inserted before another.")"""

NEW_TYPESCRIPT_FUNCTION = """function newInsertedFunction(): void {
    console.log("This is a new function inserted before another.");
}"""


NEW_PYTHON_VARIABLE = 'new_module_var = "Inserted after typed_module_var"'

NEW_TYPESCRIPT_FUNCTION_AFTER = """function newFunctionAfterClass(): void {
    console.log("This function is after DemoClass.");
}"""


class InsertBeforeSymbolTest(EditingTest):
    def __init__(self, language: Language, rel_path: str, symbol_name: str, new_content: str):
        super().__init__(language, rel_path)
        self.symbol_name = symbol_name
        self.new_content = new_content

    def _apply_edit(self, symbol_manager: SymbolManager) -> None:
        symbol_manager.insert_before_symbol(self.symbol_name, self.rel_path, self.new_content)

    def _test_diff(self, code_diff: CodeDiff) -> None:
        assert code_diff.original_content != code_diff.modified_content
        actual_added_lines = [line.strip() for _, line in code_diff.added_lines if line.strip()]
        normalized_expected_added_lines = [line.strip() for line in self.new_content.splitlines() if line.strip()]
        assert actual_added_lines == normalized_expected_added_lines


@pytest.mark.parametrize(
    "test_case",
    [
        InsertBeforeSymbolTest(
            Language.PYTHON,
            PYTHON_TEST_REL_FILE_PATH,
            "typed_module_var",
            NEW_PYTHON_VARIABLE,
        ),
        InsertBeforeSymbolTest(
            Language.PYTHON,
            PYTHON_TEST_REL_FILE_PATH,
            "use_module_variables",
            NEW_PYTHON_FUNCTION,
        ),
        InsertBeforeSymbolTest(
            Language.TYPESCRIPT,
            TYPESCRIPT_TEST_FILE,
            "DemoClass",
            NEW_TYPESCRIPT_FUNCTION_AFTER,
        ),
        InsertBeforeSymbolTest(
            Language.TYPESCRIPT,
            TYPESCRIPT_TEST_FILE,
            "helperFunction",
            NEW_TYPESCRIPT_FUNCTION,
        ),
    ],
)
def test_insert_before_symbol(test_case):
    test_case.run_test()


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


class ReplaceBodyTest(EditingTest):
    def __init__(
        self, language: Language, rel_path: str, symbol_name: str, new_body: str, expected_removed_lines: str, expected_added_lines: str
    ):
        super().__init__(language, rel_path)
        self.symbol_name = symbol_name
        self.new_body = new_body

    def _apply_edit(self, symbol_manager: SymbolManager) -> None:
        symbol_manager.replace_body(self.symbol_name, self.rel_path, self.new_body)

    def _test_diff(self, code_diff: CodeDiff) -> None:
        assert code_diff.original_content != code_diff.modified_content
        # TODO test properly


@pytest.mark.parametrize(
    "test_case",
    [
        # TODO add tests
    ],
)
def test_replace_body(test_case: ReplaceBodyTest):
    test_case.run_test()
