import argparse
import os
from .agent import OptiSpark

def main():
    parser = argparse.ArgumentParser(description="OptiSpark: Autonomous PySpark Optimization Agent")
    parser.add_argument("command", choices=["analyze"], help="Action to perform")
    parser.add_argument("--log-dir", required=True, help="Path to Spark event logs")
    
    args = parser.parse_args()

    if args.command == "analyze":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("❌ Error: GEMINI_API_KEY environment variable not set.")
            return
            
        print(f"Starting OptiSpark analysis on logs in: {args.log_dir}")
        agent = OptiSpark(log_dir=args.log_dir, api_key=api_key)
        
        # For CLI usage, we might not have the target DataFrame loaded in memory
        agent.optimize(target_df=None)

if __name__ == "__main__":
    main()