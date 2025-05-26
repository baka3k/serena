import os
import queue
import socket
import threading

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sensai.util import logging

from serena import serena_root_path

log = logging.getLogger(__name__)


class MemoryLogHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level=level)
        self.setFormatter(logging.Formatter(logging.LOG_DEFAULT_FORMAT))
        self._log_buffer = LogBuffer()
        self._log_queue = queue.Queue()
        self._stop_event = threading.Event()

        # start background thread to process logs
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()

    def emit(self, record):
        msg = self.format(record)
        self._log_queue.put_nowait(msg)

    def _process_queue(self):
        while not self._stop_event.is_set():
            try:
                msg = self._log_queue.get(timeout=1)
                self._log_buffer.append(msg)
                self._log_queue.task_done()
            except queue.Empty:
                continue

    def get_log_messages(self) -> list[str]:
        return self._log_buffer.logs


class LogBuffer:
    def __init__(self):
        self.logs = []

    def append(self, msg: str):
        self.logs.append(msg)


class RequestLog(BaseModel):
    start_idx: int = 0


class ResponseLog(BaseModel):
    messages: list[str]
    max_idx: int


class ResponseToolNames(BaseModel):
    tool_names: list[str]


class SerenaDashboardAPI:
    log = logging.getLogger(__qualname__)

    def __init__(self, memory_log_handler: MemoryLogHandler, tool_names: list[str]):
        self._memory_log_handler = memory_log_handler
        self._tool_names = tool_names
        self._app = FastAPI(title="Serena Dashboard")
        self._setup_routes()

    def _setup_routes(self):
        static_dir = os.path.join(serena_root_path(), "dashboard")
        self._app.mount("/dashboard", StaticFiles(directory=static_dir), name="dashboard")

        self._app.add_api_route("/get_log_messages", self._get_log_messages, methods=["POST"], response_model=ResponseLog)
        self._app.add_api_route("/get_tool_names", self._get_tool_names, methods=["GET"], response_model=ResponseToolNames)

    async def _get_log_messages(self, request: RequestLog) -> ResponseLog:
        all_messages = self._memory_log_handler.get_log_messages()
        requested_messages = all_messages[request.start_idx :] if request.start_idx <= len(all_messages) else []
        return ResponseLog(messages=requested_messages, max_idx=len(all_messages) - 1)

    async def _get_tool_names(self) -> ResponseToolNames:
        return ResponseToolNames(tool_names=self._tool_names)

    @staticmethod
    def _find_first_free_port(start_port: int) -> int:
        port = start_port
        while port <= 65535:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind(("localhost", port))
                    return port
            except OSError:
                port += 1

        raise RuntimeError(f"No free ports found starting from {start_port}")

    def run(self, host="127.0.0.1", port=0x5EDA):
        uvicorn.run(self._app, host=host, port=port, workers=1, log_config=None, log_level="critical")
        return port

    def run_in_thread(self) -> tuple[threading.Thread, int]:
        port = self._find_first_free_port(0x5EDA)
        thread = threading.Thread(target=lambda: self.run(port=port), daemon=True)
        thread.start()
        return thread, port
