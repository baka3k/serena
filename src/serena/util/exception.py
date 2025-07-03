import sys
from logging import Logger

from sensai.util import logging
from sensai.util.logging import FallbackHandler

from serena.agent import log


def show_fatal_exception_safe(e: Exception) -> None:
    """
    Shows the given exception in the GUI log viewer on the main thread and ensures that the exception is logged or at
    least printed to stderr.
    """
    # Make sure the error is logged (adding a fallback handler which writes to stderr in case there is no other handler)
    fallback_handler = FallbackHandler(logging.StreamHandler(sys.stderr))
    Logger.root.addHandler(fallback_handler)
    log.error(f"Fatal exception: {e}", exc_info=e)

    # attempt to show the error in the GUI
    try:
        # NOTE: The import can fail on macOS if Tk is not available (depends on Python interpreter installation, which uv
        #   used as a base); while tkinter as such is always available, its dependencies can be unavailable on macOS.
        from serena.gui_log_viewer import show_fatal_exception

        show_fatal_exception(e)
    except:
        pass
