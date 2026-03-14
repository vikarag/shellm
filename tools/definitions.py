"""Tool definitions for SheLLM agents."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "delegate_websearch",
            "description": "Search the web for current information, facts, or real-time data. Delegates to a dedicated web search agent (GPT-5 Mini with live web search).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_image",
            "description": "Analyze an image using the vision agent. Send base64-encoded image data and a prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_b64": {"type": "string", "description": "Base64-encoded image data"},
                    "prompt": {"type": "string", "description": "What to analyze in the image"},
                },
                "required": ["image_b64", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_research",
            "description": "Conduct academic or in-depth research on a topic. Uses a research-specialized agent with web search and academic focus.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The research topic or question"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_reason",
            "description": "Delegate a complex reasoning task to the deep reasoning agent (DeepSeek Reasoner). Use for planning, analysis, math, logic, and multi-step problems that benefit from chain-of-thought reasoning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The reasoning task or problem"},
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the project directory (~/shellm/) with line numbers. Can read source code, configs, etc. Use offset and limit for large files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path within the project directory (e.g. 'base_chat.py', 'workspace/notes.txt')"},
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
            "description": "Index a document for semantic search. Chunks the text, generates embeddings, and stores for later retrieval. Use when the user wants to save a document for future Q&A.",
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
            "description": "Search indexed documents by semantic similarity. Returns the most relevant chunks. Use when the user asks about previously indexed documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "top_k": {"type": "integer", "description": "Number of results to return (default: 5)"},
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
                    "schedule": {"type": "string", "description": "Cron schedule expression, e.g. '0 9 * * *' for daily at 9am, '*/5 * * * *' for every 5 minutes"},
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
            "description": "Delete a cron job by its index number (from cron_list). The user will be asked to confirm before deletion.",
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
            "description": "Execute a shell command on the system. The user will be asked to confirm before execution. Dangerous commands (rm -rf, shutdown, etc.) are automatically blocked. Use for tasks like sending emails, scheduling with 'at', checking system info, file operations, etc.",
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
            "description": "Save something to shared memory. Use this to remember user preferences, important facts, task results, or anything that should persist across sessions. All models share this memory.",
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
            "description": "Read past chat conversation logs. Returns recent chat history across all models and sessions. Use to recall what was discussed previously, find past answers, or check conversation context. Supports filtering by keyword and model name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "last_n": {"type": "integer", "description": "Number of recent entries to return (default 10, max 50)"},
                    "keyword": {"type": "string", "description": "Optional keyword to filter logs (searches user input and assistant response)"},
                    "model_filter": {"type": "string", "description": "Optional model name to filter by (e.g. 'gpt-5-mini', 'deepseek-chat')"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_file",
            "description": "Send a file to the user via Telegram. Use this to deliver files you created "
                           "(images, documents, scripts, etc.) directly in the chat. The file must exist "
                           "in workspace/ or be an absolute path. For images (png/jpg/gif/webp), sends as "
                           "a photo; otherwise sends as a document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path — relative to workspace/ (e.g. 'st_kitts_flag.png') or absolute"},
                    "caption": {"type": "string", "description": "Optional caption to send with the file"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mcp_list_servers",
            "description": "List all configured MCP servers and their connection status.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mcp_list_tools",
            "description": "List tools available from MCP servers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "server": {"type": "string", "description": "Server name to filter (optional)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_task",
            "description": "Schedule a delayed task. Supports sending a Telegram message or running a shell command at a future time. Use delay_minutes for relative scheduling or scheduled_at for absolute time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_type": {"type": "string", "enum": ["telegram_message", "shell_command"], "description": "Type of task to schedule"},
                    "payload": {"type": "object", "description": "Task data. For telegram_message: {\"message\": \"...\"}. For shell_command: {\"command\": \"...\"}. chat_id is auto-injected for Telegram."},
                    "delay_minutes": {"type": "number", "description": "Minutes from now to execute (e.g. 60 for 1 hour, 1800 for 30 hours)"},
                    "scheduled_at": {"type": "string", "description": "Absolute time in 'YYYY-MM-DD HH:MM:SS' format (KST). Alternative to delay_minutes."},
                },
                "required": ["task_type", "payload"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_scheduled_tasks",
            "description": "List scheduled tasks. Filter by status: pending, done, failed, cancelled.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "description": "Filter by status (default: pending)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_scheduled_task",
            "description": "Cancel a pending scheduled task by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer", "description": "Task ID to cancel (from list_scheduled_tasks)"},
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "report_progress",
            "description": "Report completion of a major step during plan execution. Call this after finishing each numbered step. A background AI will summarize and notify the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "step_number": {"type": "integer", "description": "Step number completed (1-based)"},
                    "total_steps": {"type": "integer", "description": "Total number of steps in the plan"},
                    "step_title": {"type": "string", "description": "Brief title of the completed step"},
                    "details": {"type": "string", "description": "What was accomplished, including key results"},
                },
                "required": ["step_number", "step_title", "details"],
            },
        },
    },
]
