import logging

from serena.mcp import mcp

log = logging.getLogger(__name__)


if __name__ == "__main__":
    log.info("Starting server")
    mcp.run()
