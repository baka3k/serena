from typing import Type


class EditingTool:
    """
    A marker class for all tools that perform editing operations on code.
    This class is used to identify tools that modify code in the project.
    Tools that inherit from this class will be excluded when the project
    is in read-only mode.
    """
    
    @classmethod
    def can_edit(cls) -> bool:
        """
        Returns True as editing tools are designed to modify code.
        This method can be overridden by subclasses if needed.
        
        :return: True if the tool can edit code, False otherwise
        """
        return True
