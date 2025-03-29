# mypy: ignore-errors
import logging
import queue
import sys
import threading
import tkinter as tk
from enum import Enum, auto


class LogLevel(Enum):
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    DEFAULT = auto()


class GuiLogViewer:
    """
    A class that creates a Tkinter GUI for displaying log messages in a separate thread.
    The log viewer supports coloring based on log levels (DEBUG, INFO, WARNING, ERROR).
    """

    def __init__(self, title="Log Viewer", width=800, height=600):
        """
        Initialize the ThreadedLogViewer.

        Args:
            title (str): The title of the window
            width (int): Initial window width
            height (int): Initial window height

        """
        self.title = title
        self.width = width
        self.height = height
        self.message_queue = queue.Queue()
        self.running = False
        self.log_thread = None

        # Define colors for different log levels
        self.log_colors = {
            LogLevel.DEBUG: "#808080",  # Gray
            LogLevel.INFO: "#000000",  # Black
            LogLevel.WARNING: "#FF8C00",  # Dark Orange
            LogLevel.ERROR: "#FF0000",  # Red
            LogLevel.DEFAULT: "#000000",  # Black
        }

    def print_status(self, s):
        print(s + "\n", file=sys.stderr)

    def start(self):
        """Start the log viewer in a separate thread."""
        if not self.running:
            self.print_status("Starting thread")
            self.running = True
            self.log_thread = threading.Thread(target=self._run_gui)
            self.log_thread.daemon = False
            self.log_thread.start()
            return True
        return False

    def stop(self):
        """Stop the log viewer."""
        if self.running:
            self.running = False
            # Add a sentinel value to the queue to signal the GUI to exit
            self.message_queue.put(None)
            return True
        return False

    def add_log(self, message):
        """
        Add a log message to the viewer.

        Args:
            message (str): The log message to display

        """
        if self.running:
            self.message_queue.put(message)
            return True
        return False

    def _determine_log_level(self, message):
        """
        Determine the log level from the message.

        Args:
            message (str): The log message

        Returns:
            LogLevel: The determined log level

        """
        message_upper = message.upper()
        if message_upper.startswith("DEBUG"):
            return LogLevel.DEBUG
        elif message_upper.startswith("INFO"):
            return LogLevel.INFO
        elif message_upper.startswith("WARNING"):
            return LogLevel.WARNING
        elif message_upper.startswith("ERROR"):
            return LogLevel.ERROR
        else:
            return LogLevel.DEFAULT

    def _process_queue(self):
        """Process messages from the queue and update the text widget."""
        try:
            while not self.message_queue.empty():
                message = self.message_queue.get_nowait()

                # Check for sentinel value to exit
                if message is None:
                    self.root.quit()
                    return

                log_level = self._determine_log_level(message)

                # Insert the message at the end of the text
                self.text_widget.configure(state=tk.NORMAL)
                self.text_widget.insert(tk.END, message + "\n", log_level.name)
                self.text_widget.configure(state=tk.DISABLED)

                # Auto-scroll to the bottom
                self.text_widget.see(tk.END)

            # Schedule to check the queue again
            if self.running:
                self.root.after(100, self._process_queue)

        except Exception as e:
            print(f"Error processing message queue: {e}", file=sys.stderr)
            if self.running:
                self.root.after(100, self._process_queue)

    def _run_gui(self):
        """Run the GUI in a separate thread."""
        try:
            self.root = tk.Tk()
            self.root.title(self.title)
            self.root.geometry(f"{self.width}x{self.height}")

            # Make the window resizable
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(0, weight=1)

            # Create frame to hold text widget and scrollbars
            frame = tk.Frame(self.root)
            frame.grid(row=0, column=0, sticky="nsew")
            frame.columnconfigure(0, weight=1)
            frame.rowconfigure(0, weight=1)

            # Create horizontal scrollbar
            h_scrollbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
            h_scrollbar.grid(row=1, column=0, sticky="ew")

            # Create vertical scrollbar
            v_scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
            v_scrollbar.grid(row=0, column=1, sticky="ns")

            # Create text widget with horizontal scrolling
            self.text_widget = tk.Text(
                frame, wrap=tk.NONE, width=self.width, height=self.height, xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set
            )
            self.text_widget.grid(row=0, column=0, sticky="nsew")
            self.text_widget.configure(state=tk.DISABLED)  # Make it read-only

            # Configure scrollbars
            h_scrollbar.config(command=self.text_widget.xview)
            v_scrollbar.config(command=self.text_widget.yview)

            # Configure tags for different log levels with appropriate colors
            for level, color in self.log_colors.items():
                self.text_widget.tag_configure(level.name, foreground=color)

            # Set up the queue processing
            self.root.after(100, self._process_queue)

            # Handle window close
            self.root.protocol("WM_DELETE_WINDOW", self.stop)

            # Start the Tkinter event loop
            self.root.mainloop()

        except Exception as e:
            print(f"Error in GUI thread: {e}", file=sys.stderr)
        finally:
            self.running = False


class GuiLogViewerHandler(logging.Handler):
    """
    A logging handler that sends log records to a ThreadedLogViewer instance.
    This handler can be integrated with Python's standard logging module
    to direct log entries to a GUI log viewer.
    """

    def __init__(
        self,
        log_viewer: GuiLogViewer,
        level=logging.NOTSET,
        format_string: str | None = "%(levelname)-5s %(asctime)-15s %(name)s:%(funcName)s:%(lineno)d - %(message)s",
    ):
        """
        Initialize the handler with a ThreadedLogViewer instance.

        Args:
            log_viewer: A ThreadedLogViewer instance that will display the logs
            level: The logging level (default: NOTSET which captures all logs)
            format_string: the format string

        """
        super().__init__(level)
        self.log_viewer = log_viewer
        self.formatter = logging.Formatter(format_string)

        # Start the log viewer if it's not already running
        if not self.log_viewer.running:
            self.log_viewer.start()

    def emit(self, record):
        """
        Emit a log record to the ThreadedLogViewer.

        Args:
            record: The log record to emit

        """
        try:
            # Format the record according to the formatter
            msg = self.format(record)

            # Convert the level name to a standard format for the viewer
            level_prefix = record.levelname

            # Add the appropriate prefix if it's not already there
            if not msg.startswith(level_prefix):
                msg = f"{level_prefix}: {msg}"

            self.log_viewer.add_log(msg)

        except Exception:
            self.handleError(record)

    def close(self):
        """
        Close the handler and optionally stop the log viewer.
        """
        # We don't automatically stop the log viewer here as it might
        # be used by other handlers or directly by the application
        super().close()

    def stop_viewer(self):
        """
        Explicitly stop the associated log viewer.
        """
        if self.log_viewer.running:
            self.log_viewer.stop()
