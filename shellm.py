#!/home/gslee/SheLLM/venv/bin/python3
"""SheLLM — minimal CLI entry point.

Usage:
    ./shellm.py                              # Interactive REPL with the default profile
    ./shellm.py --agent shellm-openrouter    # Pick a specific model profile
    ./shellm.py --image foo.png "describe"   # One-shot with an image
    ./shellm.py --list-agents                # List configured profiles
    ./shellm.py --delegate <uuid>            # Process a delegated task from Hermes

The default profile is read from the top-level `default:` field in
agent_config.yaml. Override it per-invocation with --agent.
"""

import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from agents.registry import AgentRegistry

DELEGATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace", ".delegate")


def _delegate_paths(uuid):
    return {
        "inbox": os.path.join(DELEGATE_DIR, "inbox", f"{uuid}.json"),
        "outbox": os.path.join(DELEGATE_DIR, "outbox", f"{uuid}.json"),
        "status": os.path.join(DELEGATE_DIR, "status", f"{uuid}.status"),
    }


def _load_task(uuid):
    paths = _delegate_paths(uuid)
    with open(paths["inbox"]) as f:
        return json.load(f)


def _write_result(uuid, result, tool_calls, error=None):
    paths = _delegate_paths(uuid)
    out = {
        "id": uuid,
        "status": "error" if error else "done",
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "answer": result if isinstance(result, str) else "",
        "tool_calls_made": len(tool_calls),
        "error": str(error) if error else None,
    }
    with open(paths["outbox"], "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def _write_status(uuid, state, detail=""):
    paths = _delegate_paths(uuid)
    with open(paths["status"], "w") as f:
        json.dump({
            "state": state,
            "timestamp": time.time(),
            "detail": detail,
        }, f)


def _run_delegation(uuid, agent):
    """Execute a delegated task and write structured result on completion."""
    task = _load_task(uuid)
    prompt = task.get("prompt", "")
    _write_status(uuid, "running", "started")

    # Inject delegation context into system prompt via a preamble message
    preamble = (
        f"[DELEGATED TASK from {task.get('parent_cwd', 'unknown')}]\n"
        f"Goal: {prompt}\n"
    )
    if task.get("staged_files"):
        preamble += "Staged files:\n"
        for name, rel_path in task["staged_files"].items():
            preamble += f"  - {name} -> {rel_path}\n"
    if task.get("constraints", {}).get("max_tool_calls"):
        preamble += f"Tool budget: {task['constraints']['max_tool_calls']}\n"

    messages = []
    # Prepend as a system-like context message
    messages.append({"role": "user", "content": preamble})

    # Hook heartbeat into agent
    original_execute = agent.execute_tool
    def hooked_execute(name, args):
        _write_status(uuid, "running", f"tool:{name}")
        return original_execute(name, args)
    agent.execute_tool = hooked_execute

    try:
        result = agent.process_prompt(prompt, messages)
        _write_result(uuid, result, agent._current_tool_calls)
        _write_status(uuid, "done", "completed")
    except Exception as e:
        _write_result(uuid, "", agent._current_tool_calls, error=e)
        _write_status(uuid, "error", str(e))
        raise


def main():
    parser = argparse.ArgumentParser(description="SheLLM — single-agent CLI")
    parser.add_argument(
        "--agent", default=None,
        help="Model profile to use (default: the `default:` field in agent_config.yaml)",
    )
    parser.add_argument(
        "--config",
        help="Path to agent_config.yaml (default: auto-detect)",
    )
    parser.add_argument(
        "--image",
        help="Path to an image to attach to the prompt (requires vision-capable profile)",
    )
    parser.add_argument(
        "--list-agents", action="store_true",
        help="List all configured profiles and exit",
    )
    parser.add_argument(
        "--delegate",
        help="Process a delegated task by UUID (reads from .delegate/inbox/)",
    )
    parser.add_argument(
        "prompt", nargs="?",
        help="One-shot prompt. If omitted, drops into the interactive REPL.",
    )
    args = parser.parse_args()

    registry = AgentRegistry.get_instance(config_path=args.config)

    if args.list_agents:
        default_name = registry.get_default_name()
        print("Configured profiles:\n")
        for name, config in registry.get_all_configs().items():
            flags = []
            if config.vision:
                flags.append("vision")
            if config.has_reasoning:
                flags.append("reasoning")
            if name == default_name:
                flags.append("default")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            print(f"  {name:<22} {config.provider:<12} {config.model}{flag_str}")
        sys.exit(0)

    agent_name = args.agent or registry.get_default_name()
    if not agent_name:
        print("Error: no profile specified and no `default:` set in agent_config.yaml", file=sys.stderr)
        sys.exit(1)

    try:
        agent = registry.get_agent(agent_name)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.delegate:
        _run_delegation(args.delegate, agent)
        sys.exit(0)

    if args.image:
        err = agent.attach_image(os.path.expanduser(args.image))
        if err:
            print(f"Error: {err}", file=sys.stderr)
            sys.exit(1)

    if args.prompt:
        agent.process_prompt(args.prompt, [])
    else:
        agent.run_interactive()


if __name__ == "__main__":
    main()
