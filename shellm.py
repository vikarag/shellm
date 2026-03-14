#!/home/gslee/SheLLM/venv/bin/python3
"""SheLLM — Unified multi-agent entry point.

Usage:
    ./shellm.py                              # Interactive CLI with shellm-chat
    ./shellm.py --telegram                   # Telegram bot with multi-agent routing
    ./shellm.py --agent shellm-reasoner      # Interactive with specific agent
    ./shellm.py --daemon stdin               # Daemon mode
    ./shellm.py --config path/to/config.yaml # Custom config
"""

import argparse
import os
import sys

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from agents.registry import AgentRegistry


def main():
    parser = argparse.ArgumentParser(description="SheLLM Multi-Agent System")
    parser.add_argument(
        "--agent", default="shellm-chat",
        help="Agent to use (default: shellm-chat)",
    )
    parser.add_argument(
        "--config",
        help="Path to agent_config.yaml (default: auto-detect)",
    )
    parser.add_argument(
        "--telegram", action="store_true",
        help="Run as Telegram bot with multi-agent routing",
    )
    parser.add_argument(
        "--daemon", choices=["stdin", "file", "socket"],
        help="Run in daemon mode (stdin, file, or socket)",
    )
    parser.add_argument("--json", action="store_true", help="JSON output (daemon mode)")
    parser.add_argument("--input", help="Input file path (daemon file mode)")
    parser.add_argument("--output", help="Output file path (daemon file mode)")
    parser.add_argument("--socket-path", help="Unix socket path (daemon socket mode)")
    parser.add_argument(
        "--list-agents", action="store_true",
        help="List all configured agents and exit",
    )
    args = parser.parse_args()

    # Initialize registry
    registry = AgentRegistry.get_instance(config_path=args.config)

    if args.list_agents:
        print("Configured agents:\n")
        for name, config in registry.get_all_configs().items():
            print(f"  {name:<25} {config.provider:<10} {config.model:<20} {config.system_role}")
        sys.exit(0)

    if args.telegram:
        from telegram_adapter import TelegramAdapter
        adapter = TelegramAdapter(registry)
        adapter.run()
    elif args.daemon:
        agent = registry.get_agent(args.agent)
        from daemon_mode import run_daemon
        run_daemon(agent, args.daemon, args)
    else:
        agent = registry.get_agent(args.agent)
        agent.run_interactive()


if __name__ == "__main__":
    main()
