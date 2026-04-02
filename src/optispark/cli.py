"""
OptiSpark CLI — Command-line interface for the OptiSpark agent.
"""

import argparse
import os
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

    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable not set.")
        print("   Set it with: export GEMINI_API_KEY='your_key_here'")
        return

    agent = OptiSpark(log_dir=args.log_dir, api_key=api_key)

    if args.command == "analyze":
        agent.optimize(target_df=None)
    elif args.command == "chat":
        agent.chat()


if __name__ == "__main__":
    main()