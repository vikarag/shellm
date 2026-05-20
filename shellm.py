#!/home/gslee/SheLLM/venv/bin/python3
"""SheLLM — minimal CLI entry point.

Usage:
    ./shellm.py                              # Interactive REPL with the default profile
    ./shellm.py --agent shellm-openrouter    # Pick a specific model profile
    ./shellm.py --image foo.png "describe"   # One-shot with an image
    ./shellm.py --list-agents                # List configured profiles

The default profile is read from the top-level `default:` field in
agent_config.yaml. Override it per-invocation with --agent.
"""

import argparse
import os
import sys

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from agents.registry import AgentRegistry


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
