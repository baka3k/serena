# Deadlock Analysis: MCP Symbolic Tools Timeout Issue

## Problem Summary
Intermittent deadlocks occur when using symbolic tools (particularly `find_symbol`) through MCP clients. Successful calls are followed by calls that hang for 240 seconds before timing out. Once a timeout occurs, the language server calls will no longer respond.

## ROOT CAUSE DISCOVERED: Asyncio Event Loop Interference

### Critical Evidence from MCP Server Logs
The smoking gun was found in the MCP server logs showing **unawaited coroutines**:

```
INFO mcp.server.lowlevel.server:_handle_message:524 - Warning: RuntimeWarning: coroutine 'LanguageServer.request_full_symbol_tree' was never awaited
```

This warning is being logged by the **MCP server's `_handle_message` method**, not SerenaAgent code, indicating coroutines are being created in the MCP server's asyncio context but never properly awaited.

### The Real Issue: Dual Asyncio Context Contamination

**Architecture Problem:**
1. **MCP Server**: Runs its own asyncio event loop to handle incoming requests
2. **SerenaAgent**: Creates its own asyncio event loop in separate thread for language serve, and even a second loop for the dashboard if so configured
3. **Conflict**: When MCP server calls SerenaAgent tools, we have **two or three asyncio contexts interacting**

**Deadlock Mechanism:**
```
MCP Server (asyncio loop A) 
  → handles tool request 
  → calls SerenaAgent.find_symbol()
  → SerenaAgent uses asyncio.run_coroutine_threadsafe()
  → Creates coroutine in language server loop (loop B)
  → BUT: Coroutine gets leaked into MCP's context and never awaited
  → Dangling coroutines accumulate in MCP server's loop
  → Eventually causes resource exhaustion/event loop blocking
  → Language server appears to "hang" but it's actually MCP loop contamination
```

### Discarded Hypotheses
- **Not a timeout issue**: Language server actually works fine in isolation
- **Not abandoned threads**: The threading mechanism works correctly
- **Not LSP deadlock**: The TypeScript language server itself isn't hanging

The actual issue is **asyncio context bleeding** between MCP server and SerenaAgent.

### Why We Can't Reproduce Outside MCP
- **Direct script calls**: No MCP server, no dual asyncio context
- **Single event loop**: Only SerenaAgent's language server loop exists  
- **No async interference**: Clean, isolated execution
- **MCP-specific**: Requires the exact async context interaction pattern

## Technical Details

### Key Files and Locations
- `src/multilspy/language_server.py:1870` - `run_coroutine_threadsafe` creates coroutines that leak to MCP context
- MCP server `_handle_message` - Where unawaited coroutine warnings appear

## Solution
**Process Isolation** is the fundamental fix:
1. **Separate processes**: MCP server, SerenaAgent and Dashboard in different processes
2. **IPC communication**: Replace direct method calls with inter-process communication
3. **Clean async boundaries**: Each process manages its own asyncio context
4. **No coroutine leakage**: Complete isolation prevents context contamination

This explains why the deadlock is **MCP-specific** and doesn't occur in direct tool execution - it's fundamentally about asyncio context contamination between the MCP server and SerenaAgent.
