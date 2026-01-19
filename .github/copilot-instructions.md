# GitHub Copilot Instructions

## Terminal Commands

### Python Execution
- **Never** use `python -c "..."` to run inline Python code
- **Always** create a real Python script file on disk and run it
- Example: Instead of `python -c "print('hello')"`, create a script file and run `python script.py`

### Shell Syntax
- **Never** use heredoc syntax (`<<EOF`, `<<'EOF'`, `<< 'END'`, etc.)
- **Always** create files using the file creation tools instead of heredoc
- If you need to pass multi-line content, create a proper file first

### Process Execution
- **Never** run processes in the background (no `&` at end of commands)
- **Always** run commands in the foreground and wait for completion
- If a process takes too long, use `timeout` command to limit execution time

## Rationale
These rules exist because:
1. Inline Python with `-c` has quoting/escaping issues in terminals
2. Heredoc syntax has been unreliable in this environment
3. Background processes make it difficult to track output and completion status
