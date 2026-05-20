"""Tool definitions for SheLLM."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web via DuckDuckGo. Returns a list of result snippets with title, URL, and a brief excerpt. Use this whenever you need current information or to discover URLs to read with fetch_page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "max_results": {"type": "integer", "description": "Number of results to return (default 5, max 20)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Fetch a specific URL and return its readable text content. Use after web_search when you have a specific URL to read, or when the user provides a link.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from workspace/ with line numbers. All file operations are sandboxed to workspace/. Use offset and limit for large files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path inside workspace/ — e.g. 'notes.txt' or 'subdir/file.py'. Use '.' for the workspace root. Never absolute paths."},
                    "offset": {"type": "integer", "description": "Start line (0-based, default 0)"},
                    "limit": {"type": "integer", "description": "Max lines to return (default 200)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or append to a file in workspace/. Creates parent directories automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path within workspace/"},
                    "content": {"type": "string", "description": "Content to write"},
                    "mode": {"type": "string", "enum": ["overwrite", "append"], "description": "Write mode (default: overwrite)"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories in workspace/ with sizes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path within workspace/ (default: root)"},
                    "recursive": {"type": "boolean", "description": "List recursively (default: false)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Regex search across files in workspace/. Returns matching lines with file paths and line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "Subdirectory to search in (default: all of workspace/)"},
                    "file_glob": {"type": "string", "description": "File glob filter, e.g. '*.py' (default: '*')"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_index",
            "description": "Index a document for semantic search. Chunks the text, generates embeddings, and stores it for later retrieval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The document text to index"},
                    "filename": {"type": "string", "description": "Name/label for the document"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags for categorization"},
                },
                "required": ["text", "filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Search indexed documents by semantic similarity. Returns the most relevant chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "top_k": {"type": "integer", "description": "Number of results to return (default 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_list",
            "description": "List all documents in the RAG index.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_delete",
            "description": "Delete a document from the RAG index by its doc_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "The document ID to delete (from rag_list)"},
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cron_list",
            "description": "List all current cron jobs for this user.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cron_create",
            "description": "Create a new scheduled cron job. The user will be asked to confirm before it is added.",
            "parameters": {
                "type": "object",
                "properties": {
                    "schedule": {"type": "string", "description": "Cron schedule expression, e.g. '0 9 * * *' for daily at 9am"},
                    "command": {"type": "string", "description": "Shell command to run"},
                },
                "required": ["schedule", "command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cron_delete",
            "description": "Delete a cron job by its index number (from cron_list).",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "description": "Index of the cron job to delete"},
                },
                "required": ["index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command. The user will be asked to confirm before execution. Dangerous commands are automatically blocked.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"},
                    "timeout": {"type": "integer", "description": "Max execution time in seconds (default 60)"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_read",
            "description": "Read the shared memory file. Returns all stored memories chronologically, or the last N entries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "last_n": {"type": "integer", "description": "Number of recent entries to return (0 = all, default 0)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": "Save something to shared memory. Use this to remember user preferences, important facts, or anything that should persist across sessions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The information to remember"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags for categorization"},
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search shared memory by keyword. Searches content and tags.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "Search term"},
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_delete",
            "description": "Delete a memory entry by its index (0-based chronological order). Use memory_read first to find the index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "description": "Index of the memory to delete"},
                },
                "required": ["index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "chat_log_read",
            "description": "Read past chat conversation logs. Supports filtering by keyword and model name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "last_n": {"type": "integer", "description": "Number of recent entries to return (default 10, max 50)"},
                    "keyword": {"type": "string", "description": "Optional keyword to filter logs"},
                    "model_filter": {"type": "string", "description": "Optional model name to filter by"},
                },
            },
        },
    },
]
