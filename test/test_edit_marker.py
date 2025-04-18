import pytest
from serena.agent import Tool, CreateTextFileTool, ReadFileTool, ProjectConfig
from serena.editingtool import EditingTool

class TestEditMarker:
    def test_edit_marker_class(self):
        """Test that EditingTool marker class exists"""
        assert EditingTool.__doc__ is not None
        assert hasattr(EditingTool, 'can_edit')
        assert EditingTool.can_edit() is True
        
    def test_tool_can_edit_method(self):
        """Test that Tool.can_edit() method works correctly"""
        # Non-editing tool should return False
        assert issubclass(ReadFileTool, Tool)
        assert not issubclass(ReadFileTool, EditingTool)
        assert not ReadFileTool.can_edit()
        
        # Editing tool should return True
        assert issubclass(CreateTextFileTool, Tool)
        assert issubclass(CreateTextFileTool, EditingTool)
        assert CreateTextFileTool.can_edit()
        
    def test_project_config_read_only(self):
        """Test that ProjectConfig has read_only property"""
        config = ProjectConfig({
            "language": "python",
            "project_root": "/tmp",
            "ignore_all_files_in_gitignore": True,
            "read_only": True
        }, "test_project")
        
        assert hasattr(config, 'read_only')
        assert config.read_only is True
        
        config = ProjectConfig({
            "language": "python",
            "project_root": "/tmp",
            "ignore_all_files_in_gitignore": True,
            # read_only not specified
        }, "test_project")
        
        assert hasattr(config, 'read_only')
        assert config.read_only is False  # Default should be False
