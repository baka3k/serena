# Read-Only Mode

Serena supports a read-only mode that prevents any modifications to the codebase while still allowing analysis and exploration. This feature is useful for:

- Reviewing and analyzing code without the risk of accidental modifications
- Allowing junior developers to explore a codebase with AI assistance but without modification privileges
- Generating documentation or explanations without changing the underlying code
- Creating proof-of-concept designs without implementing them

## Enabling Read-Only Mode

To enable read-only mode, set the `read_only` flag to `true` in your project configuration file:

```yaml
# whether the project is in read-only mode
read_only: true
```

## How It Works

When read-only mode is enabled:

1. All tools that perform editing operations are automatically disabled
2. Any attempt to use an editing tool will result in an error message
3. All non-editing tools remain fully functional

The editing tools that get disabled include:
- `create_text_file`: Creating or overwriting files
- `replace_symbol_body`: Replacing code symbol definitions
- `insert_after_symbol`: Inserting code after symbols
- `insert_before_symbol`: Inserting code before symbols
- `delete_lines`: Deleting lines from files
- `replace_lines`: Replacing lines in files
- `insert_at_line`: Inserting content at specific lines
- `execute_shell_command`: Executing shell commands that could potentially modify files
- `insert_at_line`: Inserting content at specific lines

## Implementation Details

Serena uses the `EditingTool` marker class to identify tools that can modify code. The base `Tool` class includes a `can_edit()` method that checks if a tool is a subclass of `EditingTool`.

When a project is configured in read-only mode:
1. During project activation, all editing tools are automatically excluded from the active tools list
2. An additional runtime check prevents any editing tools from being executed, even if they somehow remain in the active tools list

This ensures that no modifications to the codebase can occur while in read-only mode.
