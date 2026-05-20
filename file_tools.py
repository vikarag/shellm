"""File tools for SheLLM — read, write, list, and search files within workspace/.

All file operations are sandboxed to workspace/. Paths from the model are
treated as relative to workspace/ regardless of whether they have a leading
slash, so `/foo`, `foo`, and `./foo` all resolve to `workspace/foo`. Paths
that try to escape via `..` get an error string back (not an exception) so
the model can recover and retry.
"""

import os
import re

WORKSPACE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")
_WORKSPACE_REAL = os.path.realpath(WORKSPACE)


def _safe_path(relative_path):
    """Resolve a path inside workspace/.

    Returns (resolved_abs_path, error_str). On success error_str is None; on
    failure resolved_abs_path is None and error_str describes the problem in
    a way the model can act on (so it gets returned as a tool result rather
    than raised as an exception).
    """
    # Treat empty / "." / "/" as the workspace root.
    if not relative_path or relative_path in (".", "/"):
        return _WORKSPACE_REAL, None
    # Strip leading slashes so "/foo" and "foo" both anchor at workspace root.
    cleaned = relative_path.lstrip("/").lstrip("\\")
    joined = os.path.join(WORKSPACE, cleaned)
    real = os.path.realpath(joined)
    if real != _WORKSPACE_REAL and not real.startswith(_WORKSPACE_REAL + os.sep):
        return None, (
            f"Path '{relative_path}' is outside workspace/. "
            "File operations are restricted to workspace/ — use '.' for its root, "
            "or a path like 'notes.txt' or 'subdir/file.py'."
        )
    return real, None


def read_file(path, offset=0, limit=200):
    """Read a file with line numbers. Returns up to `limit` lines starting from `offset`."""
    real, err = _safe_path(path)
    if err:
        return err
    if not os.path.isfile(real):
        return f"File not found: {path}"
    try:
        with open(real, "r", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        return f"Error reading file: {e}"

    total = len(lines)
    start = max(0, offset)
    end = min(total, start + limit)
    selected = lines[start:end]

    numbered = []
    for i, line in enumerate(selected, start=start + 1):
        numbered.append(f"{i:4d} | {line.rstrip()}")

    header = f"File: {path} ({total} lines total, showing {start + 1}-{end})"
    return header + "\n" + "\n".join(numbered)


def write_file(path, content, mode="overwrite"):
    """Write or append to a file. Creates parent directories if needed."""
    real, err = _safe_path(path)
    if err:
        return err
    if real == _WORKSPACE_REAL:
        return "Refusing to write: target is the workspace root itself. Pass a filename."
    os.makedirs(os.path.dirname(real), exist_ok=True)
    write_mode = "a" if mode == "append" else "w"
    try:
        with open(real, write_mode) as f:
            f.write(content)
        action = "Appended to" if mode == "append" else "Wrote"
        size = os.path.getsize(real)
        return f"{action} {path} ({size} bytes)"
    except Exception as e:
        return f"Error writing file: {e}"


def list_directory(path=".", recursive=False):
    """List files in a directory with sizes. Caps at 200 entries."""
    real, err = _safe_path(path)
    if err:
        return err
    if not os.path.isdir(real):
        return f"Directory not found: {path}"

    entries = []
    cap = 200

    try:
        if recursive:
            for root, dirs, files in os.walk(real):
                for name in files:
                    if len(entries) >= cap:
                        break
                    full = os.path.join(root, name)
                    rel = os.path.relpath(full, WORKSPACE)
                    size = os.path.getsize(full)
                    entries.append(f"  {rel}  ({_fmt_size(size)})")
                if len(entries) >= cap:
                    break
        else:
            for name in sorted(os.listdir(real)):
                if len(entries) >= cap:
                    break
                full = os.path.join(real, name)
                rel = os.path.relpath(full, WORKSPACE)
                is_dir = os.path.isdir(full)
                if is_dir:
                    entries.append(f"  {rel}/")
                else:
                    size = os.path.getsize(full)
                    entries.append(f"  {rel}  ({_fmt_size(size)})")
    except Exception as e:
        return f"Error listing directory: {e}"

    rel_path = os.path.relpath(real, WORKSPACE) if real != WORKSPACE else "."
    header = f"Directory: {rel_path}/ ({len(entries)} items)"
    if len(entries) >= cap:
        header += f" [capped at {cap}]"
    return header + "\n" + "\n".join(entries) if entries else header + "\n  (empty)"


def search_files(pattern, path=".", file_glob="*"):
    """Regex search across files in workspace. Returns up to 50 matches."""
    import fnmatch

    real, err = _safe_path(path)
    if err:
        return err
    if not os.path.isdir(real):
        return f"Directory not found: {path}"

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Invalid regex: {e}"

    matches = []
    cap = 50

    for root, dirs, files in os.walk(real):
        for name in files:
            if not fnmatch.fnmatch(name, file_glob):
                continue
            full = os.path.join(root, name)
            rel = os.path.relpath(full, WORKSPACE)
            try:
                with open(full, "r", errors="replace") as f:
                    for lineno, line in enumerate(f, 1):
                        if regex.search(line):
                            matches.append(f"  {rel}:{lineno}: {line.rstrip()[:120]}")
                            if len(matches) >= cap:
                                break
            except Exception:
                continue
            if len(matches) >= cap:
                break
        if len(matches) >= cap:
            break

    header = f"Search: /{pattern}/ in {path} ({len(matches)} matches)"
    if len(matches) >= cap:
        header += f" [capped at {cap}]"
    return header + "\n" + "\n".join(matches) if matches else header + "\n  No matches found."


def _fmt_size(size):
    """Format file size for display."""
    if size < 1024:
        return f"{size}B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    else:
        return f"{size / (1024 * 1024):.1f}MB"
