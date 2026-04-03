"""
OptiSpark CLI — Command-line interface for the OptiSpark agent.
"""

import argparse
from .agent import OptiSpark


def main():
    parser = argparse.ArgumentParser(
        description="⚡ OptiSpark: Autonomous PySpark Optimization Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  optispark analyze --log-dir /path/to/spark/logs
  optispark chat --log-dir /path/to/spark/logs
        """,
    )
    parser.add_argument(
        "command",
        choices=["analyze", "chat"],
        help="Action to perform: 'analyze' for one-shot, 'chat' for interactive REPL",
    )
    parser.add_argument(
        "--log-dir",
        required=True,
        help="Path to Spark event logs directory",
    )
    parser.add_argument(
        "--server-url",
        default=None,
        help="Custom OptiSpark backend URL (uses default if not set)",
    )

    args = parser.parse_args()

    agent = OptiSpark(log_dir=args.log_dir, server_url=args.server_url)

    if args.command == "analyze":
        agent.optimize(target_df=None)
    elif args.command == "chat":
        agent.chat()


if __name__ == "__main__":
    main()