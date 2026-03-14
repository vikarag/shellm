#!/home/gslee/shellm/venv/bin/python3
"""Daemon mode for SheLLM agents -- stdin pipe, file, and Unix socket modes."""

import json
import os
import socket
import sys
import threading


def run_daemon(client, mode, args):
    """Dispatch to the appropriate daemon mode.

    Args:
        client: A BaseAgent instance with process_prompt, MODEL, _silent, _mode
        mode: One of 'stdin', 'file', 'socket'
        args: Parsed argparse namespace
    """
    from task_scheduler import TaskScheduler
    TaskScheduler.get_instance().start()

    client._silent = True
    client._mode = f"daemon-{mode}"
    if mode == "stdin":
        _run_stdin(client, use_json=args.json)
    elif mode == "file":
        _run_file(client, args.input, args.output, use_json=args.json)
    elif mode == "socket":
        sock_path = args.socket_path or f"/tmp/{client.MODEL}.sock"
        _run_socket(client, sock_path)


def _format_response(answer, model, use_json=False):
    """Format a response for daemon output."""
    if use_json:
        return json.dumps({"response": answer or "", "model": model}, ensure_ascii=False)
    return answer or ""


def _run_stdin(client, use_json=False):
    """Read prompts from stdin, write responses to stdout.

    Usage:
        echo "What is 2+2?" | ./gpt5mini_chat.py --daemon stdin
        echo "What is 2+2?" | ./gpt5mini_chat.py --daemon stdin --json
    """
    messages = []
    for line in sys.stdin:
        prompt = line.strip()
        if not prompt:
            continue
        answer = client.process_prompt(prompt, messages)
        output = _format_response(answer, client.MODEL, use_json)
        print(output, flush=True)


def _run_file(client, input_path, output_path, use_json=False):
    """Read prompts from a file, write responses to output file.

    Usage:
        ./gpt5mini_chat.py --daemon file --input prompts.txt --output responses.txt
    """
    if not input_path:
        print("Error: --input is required for file mode", file=sys.stderr)
        sys.exit(1)

    output_path = output_path or (input_path.rsplit(".", 1)[0] + "_responses.txt")

    messages = []
    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            prompt = line.strip()
            if not prompt:
                continue
            answer = client.process_prompt(prompt, messages)
            output = _format_response(answer, client.MODEL, use_json)
            f_out.write(output + "\n")
            f_out.flush()

    print(f"Responses written to {output_path}", file=sys.stderr)


def _handle_socket_client(conn, client, sessions):
    """Handle a single socket client connection."""
    try:
        data = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
            # Try to parse complete JSON messages
            try:
                request = json.loads(data.decode())
            except json.JSONDecodeError:
                continue

            prompt = request.get("prompt", "").strip()
            session_id = request.get("session", "default")

            if not prompt:
                response = {"error": "Empty prompt", "model": client.MODEL}
            elif prompt.lower() == "clear":
                sessions.pop(session_id, None)
                response = {"response": "Session cleared", "model": client.MODEL}
            else:
                if session_id not in sessions:
                    sessions[session_id] = []
                messages = sessions[session_id]
                answer = client.process_prompt(prompt, messages)
                response = {"response": answer or "", "model": client.MODEL}

            conn.sendall((json.dumps(response, ensure_ascii=False) + "\n").encode())
            data = b""
    except Exception as e:
        try:
            conn.sendall(json.dumps({"error": str(e)}).encode())
        except Exception:
            pass
    finally:
        conn.close()


def _run_socket(client, socket_path):
    """Run a Unix domain socket server for concurrent access.

    Usage:
        ./gpt5mini_chat.py --daemon socket --socket-path /tmp/gpt5mini.sock

    Protocol:
        Send: {"prompt": "...", "session": "default"}
        Recv: {"response": "...", "model": "..."}
    """
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    sessions = {}
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(5)
    print(f"Listening on {socket_path} (model: {client.MODEL})", file=sys.stderr)

    try:
        while True:
            conn, _ = server.accept()
            thread = threading.Thread(
                target=_handle_socket_client, args=(conn, client, sessions),
                daemon=True,
            )
            thread.start()
    except KeyboardInterrupt:
        print("\nShutting down...", file=sys.stderr)
    finally:
        server.close()
        if os.path.exists(socket_path):
            os.unlink(socket_path)
