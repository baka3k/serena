"""
Process-isolated SerenaAgent to prevent asyncio event loop contamination between MCP server and language server.

This module provides:
1. ProcessIsolatedSerenaAgent - A wrapper that runs SerenaAgent in a separate process
2. SerenaAgentWorker - The worker process that hosts the actual SerenaAgent
3. JSON-RPC based IPC for communication between processes
"""

import logging
import multiprocessing
import queue
import threading
import time
import traceback
from typing import Any, Self

from mcp.server.fastmcp.utilities.func_metadata import FuncMetadata

from serena.agent import SerenaAgent, SerenaConfig, SerenaConfigBase, Tool, ToolInterface, ToolRegistry

log = logging.getLogger(__name__)


class SerenaAgentWorker:
    """Worker process that hosts the actual SerenaAgent."""

    def __init__(self, request_queue: multiprocessing.Queue, response_queue: multiprocessing.Queue):
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.agent: SerenaAgent | None = None
        self.running = True

    def run(self) -> None:
        """Main worker loop - runs in separate process."""
        try:
            log.info("SerenaAgent worker process started")

            while self.running:
                try:
                    # Wait for requests with timeout to allow clean shutdown
                    request = self.request_queue.get(timeout=1.0)
                    if request is None:  # Shutdown signal
                        break

                    response = self._handle_request(request)
                    self.response_queue.put(response)

                except queue.Empty:
                    continue
                except Exception as e:
                    log.error(f"Error in worker process: {e}")
                    response = {
                        "id": getattr(request, "id", None),
                        "error": f"Worker process error: {e!s}",
                        "traceback": traceback.format_exc(),
                    }
                    self.response_queue.put(response)

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
            request_id = request["id"]

            if method == "initialize":
                return self._initialize(request_id, params)
            elif method == "tool_call":
                return self._tool_call(request_id, params)
            elif method == "get_active_tool_names":
                return self._get_active_tool_names(request_id)
            elif method == "is_language_server_running":
                return self._is_language_server_running(request_id)
            elif method == "reset_language_server":
                return self._reset_language_server(request_id)
            elif method == "get_exposed_tool_names":
                return self._get_exposed_tool_names(request_id)
            else:
                return {"id": request_id, "error": f"Unknown method: {method}"}

        except Exception as e:
            return {"id": request.get("id"), "error": str(e), "traceback": traceback.format_exc()}

    def _initialize(self, request_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Initialize the SerenaAgent."""
        try:
            config_dict = params["config"]
            serena_config = SerenaConfig.from_dict(config_dict)

            self.agent = SerenaAgent(serena_config=serena_config)

            return {"id": request_id, "result": "SerenaAgent initialized successfully"}
        except Exception as e:
            return {"id": request_id, "error": f"Failed to initialize SerenaAgent: {e!s}", "traceback": traceback.format_exc()}

    def _tool_call(self, request_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool call."""
        if self.agent is None:
            return {"id": request_id, "error": "SerenaAgent not initialized"}

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
                return {"id": request_id, "error": f"Tool '{tool_name}' not found or not active"}

            # Execute the tool
            result = tool.apply_ex(**tool_params)

            return {"id": request_id, "result": result}

        except Exception as e:
            return {"id": request_id, "error": str(e), "traceback": traceback.format_exc()}

    def _get_active_tool_names(self, request_id: str) -> dict[str, Any]:
        """Get list of active tool names."""
        if self.agent is None:
            return {"id": request_id, "error": "SerenaAgent not initialized"}

        try:
            tool_names = self.agent.get_active_tool_names()
            return {"id": request_id, "result": tool_names}
        except Exception as e:
            return {"id": request_id, "error": str(e), "traceback": traceback.format_exc()}

    def _is_language_server_running(self, request_id: str) -> dict[str, Any]:
        """Check if language server is running."""
        if self.agent is None:
            return {"id": request_id, "error": "SerenaAgent not initialized"}

        try:
            is_running = self.agent.is_language_server_running()
            return {"id": request_id, "result": is_running}
        except Exception as e:
            return {"id": request_id, "error": str(e), "traceback": traceback.format_exc()}

    def _reset_language_server(self, request_id: str) -> dict[str, Any]:
        """Reset the language server."""
        if self.agent is None:
            return {"id": request_id, "error": "SerenaAgent not initialized"}

        try:
            self.agent.reset_language_server()
            return {"id": request_id, "result": "Language server reset successfully"}
        except Exception as e:
            return {"id": request_id, "error": str(e), "traceback": traceback.format_exc()}

    def _get_exposed_tool_names(self, request_id: str) -> dict[str, Any]:
        """Get exposed tool names for MCP tool creation."""
        if self.agent is None:
            return {"id": request_id, "error": "SerenaAgent not initialized"}

        try:
            tool_instances = self.agent.get_exposed_tool_instances()
            # Return only tool names - metadata will be reconstructed from ToolRegistry
            tool_names = [tool.get_name_from_cls() for tool in tool_instances]
            return {"id": request_id, "result": tool_names}
        except Exception as e:
            return {"id": request_id, "error": str(e), "traceback": traceback.format_exc()}

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self.agent is not None:
            try:
                if self.agent.is_language_server_running() and self.agent.language_server is not None:
                    self.agent.language_server.stop()
            except Exception as e:
                log.error(f"Error stopping language server: {e}")
            self.agent = None


class ProcessIsolatedSerenaAgent:
    """Process-isolated wrapper for SerenaAgent that prevents asyncio contamination."""

    def __init__(self, serena_config: SerenaConfigBase):
        self.serena_config = serena_config
        self.process: multiprocessing.Process | None = None
        self.request_queue: multiprocessing.Queue | None = None
        self.response_queue: multiprocessing.Queue | None = None
        self.request_id_counter = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the worker process."""
        if self.process is not None:
            raise RuntimeError("ProcessIsolatedSerenaAgent already started")

        log.info("Starting process-isolated SerenaAgent")

        # Create communication queues
        self.request_queue = multiprocessing.Queue()
        self.response_queue = multiprocessing.Queue()

        # Create and start worker process
        worker = SerenaAgentWorker(self.request_queue, self.response_queue)
        self.process = multiprocessing.Process(target=worker.run)
        self.process.start()

        # Initialize the agent in the worker process
        try:
            self._make_request_with_result("initialize", {"config": self.serena_config.to_dict()})
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
            # Send shutdown signal
            if self.request_queue is not None:
                self.request_queue.put(None)

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
            self.request_queue = None
            self.response_queue = None

        log.info("Process-isolated SerenaAgent stopped")

    def _make_request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make a request to the worker process."""
        if self.process is None or not self.process.is_alive():
            raise RuntimeError("Worker process is not running")

        with self._lock:
            self.request_id_counter += 1
            request_id = str(self.request_id_counter)

        request = {"id": request_id, "method": method, "params": params or {}}

        # Send request
        if self.request_queue is not None:
            self.request_queue.put(request)

        # Wait for response
        timeout = self.serena_config.tool_timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                if self.response_queue is not None:
                    response = self.response_queue.get(timeout=1.0)
                else:
                    raise RuntimeError("Response queue is not initialized")
                if response["id"] == request_id:
                    return response
                else:
                    # Put back response for different request
                    if self.response_queue is not None:
                        self.response_queue.put(response)
            except queue.Empty:
                continue

        raise TimeoutError(f"Request {request_id} timed out after {timeout} seconds")

    def _make_request_with_result(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Make a request and return the result, raising an exception if there's an error."""
        response = self._make_request(method, params)
        if "error" in response:
            raise RuntimeError(f"Request {method} failed: {response['error']}")
        return response["result"]

    def tool_call(self, tool_name: str, **tool_params: Any) -> str:
        """Call a tool in the worker process."""
        return self._make_request_with_result("tool_call", {"tool_name": tool_name, "tool_params": tool_params})

    def get_active_tool_names(self) -> list[str]:
        """Get list of active tool names."""
        return self._make_request_with_result("get_active_tool_names")

    def is_language_server_running(self) -> bool:
        """Check if language server is running."""
        return self._make_request_with_result("is_language_server_running")

    def reset_language_server(self) -> None:
        """Reset the language server."""
        self._make_request_with_result("reset_language_server")

    def get_exposed_tool_names(self) -> list[str]:
        """Get tool names for MCP tool creation."""
        return self._make_request_with_result("get_exposed_tool_names")

    def get_exposed_tool_instances(self) -> list[ToolInterface]:
        """Get exposed tool instances for MCP tool creation."""
        tool_names = self.get_exposed_tool_names()
        return [ProcessIsolatedTool(self, tool_name) for tool_name in tool_names]

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
