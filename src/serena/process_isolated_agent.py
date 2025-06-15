"""
Process-isolated SerenaAgent to prevent asyncio event loop contamination between MCP server and language server.

This module provides:
1. ProcessIsolatedSerenaAgent - A wrapper that runs SerenaAgent in a separate process
2. SerenaAgentWorker - The worker process that hosts the actual SerenaAgent
3. JSON-RPC based IPC for communication between processes
"""

import asyncio
import contextlib
import logging
import multiprocessing
import signal
import traceback
from collections.abc import Callable
from enum import StrEnum
from logging.handlers import QueueHandler
from multiprocessing.connection import Connection
from typing import Any, Literal, Self

import uvicorn
from mcp.server.fastmcp.utilities.func_metadata import FuncMetadata
from sensai.util.logging import LOG_DEFAULT_FORMAT

from serena.agent import SerenaAgent, SerenaConfig, SerenaConfigBase, Tool, ToolInterface, ToolRegistry
from serena.config import SerenaAgentContext, SerenaAgentMode
from serena.dashboard import MemoryLogHandler, SerenaDashboardAPI

log = logging.getLogger(__name__)

_global_log_q: "multiprocessing.Queue[str]" = multiprocessing.Queue()


# Global shutdown function that can be called by the dashboard
class _GlobalShutdownRegistry:
    """Registry for global shutdown function."""

    def __init__(self) -> None:
        self.shutdown_func: Callable[[], None] | None = None

    def set_shutdown_func(self, func: Callable[[], None]) -> None:
        """Set the global shutdown function."""
        self.shutdown_func = func

    def call_shutdown(self) -> None:
        """Call the global shutdown function if it exists."""
        if self.shutdown_func:
            self.shutdown_func()


_shutdown_registry = _GlobalShutdownRegistry()


def set_global_shutdown_func(func: Callable[[], None]) -> None:
    """Set the global shutdown function."""
    _shutdown_registry.set_shutdown_func(func)


def call_global_shutdown() -> None:
    """Call the global shutdown function if it exists."""
    _shutdown_registry.call_shutdown()


class ProcessIsolatedDashboard:
    """Process-isolated dashboard to prevent asyncio contamination with MCP server and worker."""

    def __init__(self, tool_names: list[str]):
        self.tool_names = tool_names
        self.process: multiprocessing.Process | None = None
        self.shutdown_queue: multiprocessing.Queue[bool] | None = None
        self.port_queue: multiprocessing.Queue[int] | None = None
        self._log_handler: ProcessDashboardLogHandler | None = None

    def start(self) -> int:
        """Start the dashboard process and return the port number."""
        if self.process is not None:
            raise RuntimeError("Dashboard process already started")

        log.info("Starting process-isolated dashboard")

        # Create communication queues
        self.shutdown_queue = multiprocessing.Queue()
        self.port_queue = multiprocessing.Queue()

        # Create and start dashboard process
        dashboard_worker = DashboardWorker(shutdown_queue=self.shutdown_queue, port_queue=self.port_queue, tool_names=self.tool_names)
        self.process = multiprocessing.Process(target=dashboard_worker.run)
        self.process.start()

        # Set up log handler to send logs to dashboard process
        logging.Logger.root.addHandler(ProcessDashboardLogHandler())

        # Wait for dashboard to start and return port
        try:
            port = self.port_queue.get(timeout=10)  # Wait up to 10 seconds for startup
            log.debug(f"Dashboard started on port {port}")
            return port
        except Exception as e:
            self.stop()
            raise RuntimeError(f"Failed to start dashboard: {e}") from e

    def stop(self) -> None:
        """Stop the dashboard process."""
        if self.process is None:
            return

        log.info("Stopping process-isolated dashboard")

        try:
            # Signal shutdown
            if self.shutdown_queue is not None:
                import contextlib

                with contextlib.suppress(Exception):
                    self.shutdown_queue.put_nowait(True)

            # Wait for process to terminate
            if self.process.is_alive():
                self.process.join(timeout=5.0)

                if self.process.is_alive():
                    log.warning("Dashboard process did not terminate gracefully, forcing termination")
                    self.process.terminate()
                    self.process.join(timeout=3.0)

                    if self.process.is_alive():
                        log.error("Dashboard process could not be terminated, killing it")
                        self.process.kill()
                        self.process.join()

        except Exception as e:
            log.error(f"Error stopping dashboard process: {e}")

        finally:
            self.process = None
            self.shutdown_queue = None
            self.port_queue = None

        log.info("Process-isolated dashboard stopped")

    def update_tool_names(self, tool_names: list[str]) -> None:
        """Update tool names (for future enhancement)."""
        self.tool_names = tool_names
        # TODO: Could send update to dashboard process if needed


class ProcessDashboardLogHandler(logging.Handler):
    """Log handler that sends log messages to dashboard process via queue."""

    def __init__(self, level: int = logging.NOTSET):
        super().__init__(level=level)
        self.setFormatter(logging.Formatter(LOG_DEFAULT_FORMAT))

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        # Non-blocking put to avoid deadlocks
        _global_log_q.put_nowait(msg)


class DashboardWorker:
    """Worker process that hosts the dashboard with its own asyncio loop."""

    def __init__(
        self,
        shutdown_queue: "multiprocessing.Queue[bool]",
        port_queue: "multiprocessing.Queue[int]",
        tool_names: list[str],
    ):
        self.shutdown_queue = shutdown_queue
        self.port_queue = port_queue
        self.tool_names = tool_names

    def run(self) -> None:
        """Main dashboard worker loop - runs in separate process."""
        try:
            log.info("Dashboard worker process started")
            asyncio.run(self._async_main())
        except Exception as e:
            log.error(f"Fatal error in dashboard worker process: {e}")
        finally:
            log.info("Dashboard worker process stopped")

    async def _async_main(self) -> None:
        """Main async loop for dashboard process."""
        log_processor_task = None
        try:
            # Set up dashboard API with shutdown callback
            dashboard_api = SerenaDashboardAPI(
                memory_log_handler=MemoryLogHandler(), tool_names=self.tool_names, shutdown_callback=self._handle_shutdown
            )

            # Start log processor task
            log_processor_task = asyncio.create_task(self._process_log_queue(dashboard_api))

            # Find free port and signal to parent
            port = dashboard_api._find_first_free_port(0x5EDA)
            self.port_queue.put(port)

            config = uvicorn.Config(app=dashboard_api._app, host="0.0.0.0", port=port, workers=1, log_config=None, log_level="critical")
            server = uvicorn.Server(config)

            # Start server and shutdown monitor concurrently
            shutdown_task = asyncio.create_task(self._monitor_shutdown())
            server_task = asyncio.create_task(server.serve())

            # Wait for either server to complete or shutdown signal
            done, pending = await asyncio.wait([server_task, shutdown_task], return_when=asyncio.FIRST_COMPLETED)
            # Cancel pending tasks
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        except Exception as e:
            log.error(f"Error in dashboard worker: {e}")
        finally:
            # Clean up
            if log_processor_task and not log_processor_task.done():
                log_processor_task.cancel()

                with contextlib.suppress(asyncio.CancelledError):
                    await log_processor_task

    @staticmethod
    async def _process_log_queue(dashboard_api: SerenaDashboardAPI) -> None:
        """Process log messages from the queue and forward to memory handler."""
        while True:
            try:
                # Check for log messages
                while not _global_log_q.empty():
                    try:
                        log_msg = _global_log_q.get_nowait()
                        record = logging.LogRecord(
                            name="forwarded", level=logging.INFO, pathname="", lineno=0, msg=log_msg, args=(), exc_info=None
                        )
                        dashboard_api.memory_log_handler.emit(record)
                    except Exception as e:
                        log.error(f"Error processing log message: {e}")
                        await asyncio.sleep(1)

                # Small delay to avoid busy waiting
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error processing log queue: {e}")
                await asyncio.sleep(1)

    async def _monitor_shutdown(self) -> None:
        """Monitor for shutdown signals."""
        while True:
            try:
                # Check for shutdown signal
                if not self.shutdown_queue.empty():
                    self.shutdown_queue.get_nowait()
                    log.info("Dashboard shutdown signal received")
                    break

                await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error monitoring shutdown: {e}")
                await asyncio.sleep(1)

    def _handle_shutdown(self) -> None:
        """Handle shutdown request from dashboard UI."""
        log.info("Dashboard UI shutdown requested")
        # Signal shutdown to the main process through the parent dashboard
        if self.shutdown_queue is not None:
            try:
                self.shutdown_queue.put_nowait(True)
                log.info("Sent shutdown signal to main process via queue")
            except Exception as e:
                log.error(f"Failed to send shutdown signal: {e}")

        # Also try the global shutdown as fallback
        try:
            call_global_shutdown()
        except Exception as e:
            log.error(f"Global shutdown failed: {e}")

        # Force exit this dashboard process
        import os
        import threading

        def delayed_exit() -> None:
            import time

            time.sleep(0.5)  # Give time for shutdown signal to be sent
            log.info("Dashboard process forcing exit")
            os._exit(0)

        threading.Thread(target=delayed_exit, daemon=True).start()


class SerenaAgentWorker:
    """Worker process that hosts the actual SerenaAgent."""

    def __init__(self, conn: Connection):
        self.conn = conn
        self.agent: SerenaAgent | None = None
        self._shutdown_requested = False

    def run(self, q: "multiprocessing.Queue[str]") -> None:
        """Main worker loop - runs in separate process."""
        qh = QueueHandler(q)
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        root.addHandler(qh)

        try:
            # Set up signal handler for clean shutdown
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)

            log.info("SerenaAgent worker process started")

            while not self._shutdown_requested:
                try:
                    # Use polling to avoid blocking indefinitely
                    if self.conn.poll(timeout=0.5):  # Poll every 500ms
                        try:
                            request = self.conn.recv()
                            if request is None:  # Explicit shutdown signal
                                break

                            response = self._handle_request(request)
                            self.conn.send(response)
                        except EOFError:
                            # Connection closed - parent process terminated
                            log.info("Connection closed, worker shutting down")
                            break
                        except (BrokenPipeError, ConnectionResetError):
                            # Connection broken, can't communicate
                            log.info("Connection broken, worker shutting down")
                            break
                        except Exception as e:
                            log.error(f"Error processing request: {e}")
                            try:
                                response = {
                                    "error": f"Worker process error: {e!s}",
                                    "traceback": traceback.format_exc(),
                                }
                                self.conn.send(response)
                            except (EOFError, BrokenPipeError, ConnectionResetError):
                                # Connection is broken, can't send error response
                                log.info("Connection broken during error response, shutting down")
                                break
                    # Continue polling if no data available

                except Exception as e:
                    log.error(f"Error in worker main loop: {e}")
                    break

        except Exception as e:
            log.error(f"Fatal error in worker process: {e}")
        finally:
            self._cleanup()
            log.info("SerenaAgent worker process stopped")

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals gracefully."""
        log.info(f"Received signal {signum}, initiating shutdown")
        self._shutdown_requested = True

    def _handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle a single request."""
        try:
            method = request["method"]
            params = request.get("params", {})

            match method:
                case self.RequestMethod.INITIALIZE:
                    return self._initialize(params)
                case self.RequestMethod.TOOL_CALL:
                    return self._tool_call(params)
                case self.RequestMethod.GET_ACTIVE_TOOL_NAMES:
                    return self._get_active_tool_names()
                case self.RequestMethod.IS_LANGUAGE_SERVER_RUNNING:
                    return self._is_language_server_running()
                case self.RequestMethod.RESET_LANGUAGE_SERVER:
                    return self._reset_language_server()
                case self.RequestMethod.GET_EXPOSED_TOOL_NAMES:
                    return self._get_exposed_tool_names()
                case self.RequestMethod.SHUTDOWN:
                    return self._shutdown()
                case _:
                    return {"error": f"Unknown method: {method}"}

        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc()}

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Initialize the SerenaAgent."""
        try:
            # Extract all possible initialization parameters
            project = params.get("project")
            project_activation_callback = params.get("project_activation_callback")
            serena_config = params.get("serena_config")
            context = params.get("context")
            modes = params.get("modes")
            enable_web_dashboard = params.get("enable_web_dashboard")
            enable_gui_log_window = params.get("enable_gui_log_window")
            log_level = params.get("log_level")
            trace_lsp_communication = params.get("trace_lsp_communication")
            tool_timeout = params.get("tool_timeout")

            # Handle legacy config_dict parameter for backward compatibility
            if "config" in params and serena_config is None:
                config_dict = params["config"]
                serena_config = SerenaConfig.from_dict(config_dict)
            elif serena_config is not None and isinstance(serena_config, dict):
                serena_config = SerenaConfig.from_dict(serena_config)

            self.agent = SerenaAgent(
                project=project,
                project_activation_callback=project_activation_callback,
                serena_config=serena_config,
                context=context,
                modes=modes,
                enable_web_dashboard=enable_web_dashboard,
                enable_gui_log_window=enable_gui_log_window,
                log_level=log_level,
                trace_lsp_communication=trace_lsp_communication,
                tool_timeout=tool_timeout,
            )

            # Set up global shutdown function for dashboard
            def shutdown_worker() -> None:
                log.info("Global shutdown function called")
                self._cleanup()
                import sys

                sys.exit(0)

            set_global_shutdown_func(shutdown_worker)

            return {"result": "SerenaAgent initialized successfully"}
        except Exception as e:
            return {"error": f"Failed to initialize SerenaAgent: {e!s}", "traceback": traceback.format_exc()}

    def _tool_call(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool call."""
        if self.agent is None:
            return {"error": "SerenaAgent not initialized"}

        try:
            tool_name = params["tool_name"]
            tool_params = params["tool_params"]

            # Get the tool by name
            tool = None
            for tool_instance in self.agent._active_tools.values():
                if tool_instance.get_name_from_cls() == tool_name:
                    tool = tool_instance
                    break

            if tool is None:
                return {"error": f"Tool '{tool_name}' not found or not active"}

            # Execute the tool
            result = tool.apply_ex(**tool_params)

            return {"result": result}

        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc()}

    def _get_active_tool_names(self) -> dict[str, Any]:
        """Get list of active tool names."""
        if self.agent is None:
            return {"error": "SerenaAgent not initialized"}

        try:
            tool_names = self.agent.get_active_tool_names()
            return {"result": tool_names}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc()}

    def _is_language_server_running(self) -> dict[str, Any]:
        """Check if language server is running."""
        if self.agent is None:
            return {"error": "SerenaAgent not initialized"}

        try:
            is_running = self.agent.is_language_server_running()
            return {"result": is_running}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc()}

    def _reset_language_server(self) -> dict[str, Any]:
        """Reset the language server."""
        if self.agent is None:
            return {"error": "SerenaAgent not initialized"}

        try:
            self.agent.reset_language_server()
            return {"result": "Language server reset successfully"}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc()}

    def _get_exposed_tool_names(self) -> dict[str, Any]:
        """Get exposed tool names for MCP tool creation."""
        if self.agent is None:
            return {"error": "SerenaAgent not initialized"}

        try:
            tool_instances = self.agent.get_exposed_tool_instances()
            # Return only tool names - metadata will be reconstructed from ToolRegistry
            tool_names = [tool.get_name_from_cls() for tool in tool_instances]
            return {"result": tool_names}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc()}

    def _shutdown(self) -> dict[str, Any]:
        """Handle shutdown request from dashboard."""
        try:
            log.info("Shutdown requested from dashboard")
            # Clean up resources before exiting
            self._cleanup()
            # Exit the worker process - this will cause the main process to detect the termination
            # and shut down gracefully through the server lifespan manager
            import sys

            sys.exit(0)
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc()}

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self.agent is not None:
            try:
                if self.agent.is_language_server_running() and self.agent.language_server is not None:
                    self.agent.language_server.stop()
            except Exception as e:
                log.error(f"Error stopping language server: {e}")
            self.agent = None

    class RequestMethod(StrEnum):
        """Enumeration of available request methods."""

        INITIALIZE = "initialize"
        TOOL_CALL = "tool_call"
        GET_ACTIVE_TOOL_NAMES = "get_active_tool_names"
        IS_LANGUAGE_SERVER_RUNNING = "is_language_server_running"
        RESET_LANGUAGE_SERVER = "reset_language_server"
        GET_EXPOSED_TOOL_NAMES = "get_exposed_tool_names"
        SHUTDOWN = "shutdown"


class ProcessIsolatedSerenaAgent:
    """Process-isolated wrapper for SerenaAgent that prevents asyncio contamination."""

    def __init__(
        self,
        project: str | None = None,
        project_activation_callback: Callable[[], None] | None = None,
        serena_config: SerenaConfigBase | None = None,
        context: SerenaAgentContext | None = None,
        modes: list[SerenaAgentMode] | None = None,
        enable_web_dashboard: bool | None = None,
        enable_gui_log_window: bool | None = None,
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None = None,
        trace_lsp_communication: bool | None = None,
        tool_timeout: float | None = None,
    ):
        # Store all initialization parameters to pass to worker
        self._init_params = {
            "project": project,
            "project_activation_callback": project_activation_callback,
            "serena_config": serena_config,
            "context": context,
            "modes": modes,
            "enable_web_dashboard": enable_web_dashboard,
            "enable_gui_log_window": enable_gui_log_window,
            "log_level": log_level,
            "trace_lsp_communication": trace_lsp_communication,
            "tool_timeout": tool_timeout,
        }

        # Keep serena_config for compatibility
        if serena_config is not None:
            self.serena_config = serena_config
        else:
            # Use the default config loader
            self.serena_config = SerenaConfig.from_config_file()
        self.process: multiprocessing.Process | None = None
        self.conn: Connection | None = None

    def start(self) -> None:
        """Start the worker process."""
        if self.process is not None:
            raise RuntimeError("ProcessIsolatedSerenaAgent already started")

        log.info("Starting process-isolated SerenaAgent")

        # Create communication pipe
        parent_conn, child_conn = multiprocessing.Pipe()
        self.conn = parent_conn

        # Create and start worker process, passing along the dashboard's queue if available
        worker = SerenaAgentWorker(child_conn)
        self.process = multiprocessing.Process(target=worker.run, args=[_global_log_q])
        self.process.start()

        # Initialize the agent in the worker process
        try:
            # Prepare initialization parameters, converting complex objects to dict if present
            init_params = self._init_params.copy()
            if init_params["serena_config"] is not None and hasattr(init_params["serena_config"], "to_dict"):
                # Convert SerenaConfigBase to dict for serialization
                serena_config_obj = init_params["serena_config"]
                init_params["serena_config"] = serena_config_obj.to_dict()  # type: ignore
            # Note: context and modes are not serializable across processes,
            # so they will be None and SerenaAgent will use defaults
            init_params["context"] = None
            init_params["modes"] = None
            # project_activation_callback cannot be serialized across processes
            init_params["project_activation_callback"] = None
            # Disable web dashboard in worker process - it will run in MCP process instead
            init_params["enable_web_dashboard"] = False

            self._make_request_with_result(SerenaAgentWorker.RequestMethod.INITIALIZE, init_params)
        except Exception as e:
            self.stop()
            raise RuntimeError(f"Failed to initialize SerenaAgent: {e}") from e

        log.info("Process-isolated SerenaAgent started successfully")

    def stop(self) -> None:
        """Stop the worker process."""
        if self.process is None:
            return

        log.info("Stopping process-isolated SerenaAgent")

        try:
            # Close connection to signal worker to shutdown
            if self.conn is not None:
                self.conn.close()

            # Wait for process to terminate with shorter timeout
            if self.process.is_alive():
                self.process.join(timeout=3.0)  # Reduced from 10s to 3s

                if self.process.is_alive():
                    log.warning("Worker process did not terminate gracefully, sending SIGTERM")
                    self.process.terminate()
                    self.process.join(timeout=2.0)  # Reduced from 5s to 2s

                    if self.process.is_alive():
                        log.error("Worker process did not respond to SIGTERM, sending SIGKILL")
                        self.process.kill()
                        self.process.join(timeout=1.0)  # Add timeout for kill as well

                        if self.process.is_alive():
                            log.error("Worker process could not be killed - forcing cleanup")

        except KeyboardInterrupt:
            log.warning("Keyboard interrupt during shutdown - forcing termination")
            if self.process and self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=1.0)
        except Exception as e:
            log.error(f"Error stopping worker process: {e}")

        finally:
            self.process = None
            self.conn = None

        log.info("Process-isolated SerenaAgent stopped")

    def _make_request(self, method: SerenaAgentWorker.RequestMethod, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make a request to the worker process."""
        if self.process is None or not self.process.is_alive():
            raise RuntimeError("Worker process is not running")

        if self.conn is None:
            raise RuntimeError("Connection is not initialized")

        request = {"method": method, "params": params or {}}

        # Send request
        try:
            self.conn.send(request)
        except (EOFError, BrokenPipeError) as e:
            raise RuntimeError("Failed to send request: worker process may have crashed") from e

        # Wait for response with timeout
        timeout = self.serena_config.tool_timeout
        if self.conn.poll(timeout):
            try:
                response = self.conn.recv()
                return response
            except (EOFError, BrokenPipeError) as e:
                raise RuntimeError("Failed to receive response: worker process may have crashed") from e
        else:
            raise TimeoutError(f"Request {method} timed out after {timeout} seconds")

    def _make_request_with_result(self, method: SerenaAgentWorker.RequestMethod, params: dict[str, Any] | None = None) -> Any:
        """Make a request and return the result, raising an exception if there's an error."""
        response = self._make_request(method, params)
        if "error" in response:
            raise RuntimeError(f"Request {method} failed: {response['error']}")
        return response["result"]

    def tool_call(self, tool_name: str, **tool_params: Any) -> str:
        """Call a tool in the worker process."""
        return self._make_request_with_result(
            SerenaAgentWorker.RequestMethod.TOOL_CALL, {"tool_name": tool_name, "tool_params": tool_params}
        )

    def get_tool(self, tool_cls: type[Tool]) -> "ProcessIsolatedTool":
        """Get a process-isolated tool that delegates to this agent."""
        tool_name = tool_cls.get_name_from_cls()
        return ProcessIsolatedTool(self, tool_name)

    def get_active_tool_names(self) -> list[str]:
        """Get list of active tool names."""
        return self._make_request_with_result(SerenaAgentWorker.RequestMethod.GET_ACTIVE_TOOL_NAMES)

    def is_language_server_running(self) -> bool:
        """Check if language server is running."""
        return self._make_request_with_result(SerenaAgentWorker.RequestMethod.IS_LANGUAGE_SERVER_RUNNING)

    def reset_language_server(self) -> None:
        """Reset the language server."""
        self._make_request_with_result(SerenaAgentWorker.RequestMethod.RESET_LANGUAGE_SERVER)

    def shutdown_from_dashboard(self) -> None:
        """Request shutdown from dashboard."""
        self._make_request_with_result(SerenaAgentWorker.RequestMethod.SHUTDOWN)

    def get_exposed_tool_names(self) -> list[str]:
        """Get tool names for MCP tool creation."""
        return self._make_request_with_result(SerenaAgentWorker.RequestMethod.GET_EXPOSED_TOOL_NAMES)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


class ProcessIsolatedTool(ToolInterface):
    """A clean tool wrapper that delegates to ProcessIsolatedSerenaAgent."""

    def __init__(self, process_agent: ProcessIsolatedSerenaAgent, tool_name: str):
        self.process_agent = process_agent
        self._tool_name = tool_name

    @property
    def _tool_class(self) -> type[Tool]:
        return ToolRegistry.get_tool_class_by_name(self._tool_name)

    def get_name(self) -> str:
        """Get the tool name for this process-isolated tool."""
        return self._tool_name

    def get_apply_docstring(self) -> str:
        """Get the docstring for the apply method."""
        # in the actual tool, this is a classmethod
        return self._tool_class.get_apply_docstring_from_cls()

    def get_apply_fn_metadata(self) -> FuncMetadata:
        """Get the metadata for the apply method."""
        # in the actual tool, this is a classmethod
        return self._tool_class.get_apply_fn_metadata_from_cls()

    def apply_ex(self, log_call: bool = True, catch_exceptions: bool = True, **kwargs: Any) -> str:
        """Apply the tool with logging and exception handling."""
        try:
            return self.process_agent.tool_call(self._tool_name, **kwargs)
        except Exception as e:
            if catch_exceptions:
                return f"Error executing tool {self._tool_name}: {e!s}"
            raise
