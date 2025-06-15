"""
Process-isolated SerenaAgent to prevent asyncio event loop contamination between MCP server and language server.

This module provides:
1. ProcessIsolatedSerenaAgent - A wrapper that runs SerenaAgent in a separate process
2. SerenaAgentWorker - The worker process that hosts the actual SerenaAgent
3. JSON-RPC based IPC for communication between processes
"""

import logging
import multiprocessing
import traceback
from collections.abc import Callable
from enum import StrEnum
from multiprocessing.connection import Connection
from typing import Any, Literal, Self

from mcp.server.fastmcp.utilities.func_metadata import FuncMetadata

from serena.agent import SerenaAgent, SerenaConfig, SerenaConfigBase, Tool, ToolInterface, ToolRegistry
from serena.config import SerenaAgentContext, SerenaAgentMode

log = logging.getLogger(__name__)


class SerenaAgentWorker:
    """Worker process that hosts the actual SerenaAgent."""

    def __init__(self, conn: Connection):
        self.conn = conn
        self.agent: SerenaAgent | None = None

    def run(self) -> None:
        """Main worker loop - runs in separate process."""
        try:
            log.info("SerenaAgent worker process started")

            while True:
                try:
                    # Block until request received or connection closed
                    request = self.conn.recv()
                    if request is None:  # Shutdown signal
                        break

                    response = self._handle_request(request)
                    self.conn.send(response)

                except EOFError:
                    # Connection closed - parent process terminated
                    log.info("Connection closed, worker shutting down")
                    break
                except Exception as e:
                    log.error(f"Error in worker process: {e}")
                    try:
                        response = {
                            "error": f"Worker process error: {e!s}",
                            "traceback": traceback.format_exc(),
                        }
                        self.conn.send(response)
                    except (EOFError, BrokenPipeError):
                        # Connection is broken, can't send error response
                        break

        except Exception as e:
            log.error(f"Fatal error in worker process: {e}")
        finally:
            self._cleanup()
            log.info("SerenaAgent worker process stopped")

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

        # Create and start worker process
        worker = SerenaAgentWorker(child_conn)
        self.process = multiprocessing.Process(target=worker.run)
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

            # Wait for process to terminate
            if self.process.is_alive():
                self.process.join(timeout=10.0)

                if self.process.is_alive():
                    log.warning("Worker process did not terminate gracefully, forcing termination")
                    self.process.terminate()
                    self.process.join(timeout=5.0)

                    if self.process.is_alive():
                        log.error("Worker process could not be terminated, killing it")
                        self.process.kill()
                        self.process.join()

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
