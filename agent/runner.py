import os
import sys
import argparse
import json
import logging

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.model_agent import StockScoutModelAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AgentRunner")

# Fix encoding for Windows terminals
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def print_banner():
    banner = """
============================================================
       STOCK SCOUT -- AUTONOMOUS MODEL AGENT
     Model Performance Monitoring & Auto-Retraining
============================================================
    """
    print(banner)

def main():
    parser = argparse.ArgumentParser(
        description="Stock Scout Autonomous Model Agent CLI Runner"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--monitor", action="store_true", help="Monitor watchlist model performance and detect drift")
    group.add_argument("--train", action="store_true", help="Train / retrain directional ML models")
    group.add_argument("--auto", action="store_true", help="Autonomous audit & auto-retrain drifted models")
    group.add_argument("--status", action="store_true", help="Display current agent state and model health")

    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated stock symbols (e.g. CDSL.NS,HDFCBANK.NS)")
    parser.add_argument("--threshold", type=float, default=55.0, help="Accuracy threshold percentage for drift detection (default: 55.0)")
    parser.add_argument("--period", type=str, default="2y", help="Historical data period for training (default: 2y)")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()
    if not args.json:
        print_banner()

    agent = StockScoutModelAgent()
    symbols_list = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None

    if args.status:
        state = agent.get_state()
        if args.json:
            print(json.dumps(state, indent=2))
        else:
            print(f"Agent Name: {state.get('agent_name')}")
            print(f"Last Run:   {state.get('last_run', 'Never')}")
            print("\nMonitored Models Status:")
            models = state.get("monitored_models", {})
            if not models:
                print("  No models monitored yet. Run with --monitor or --auto to populate.")
            for sym, data in models.items():
                print(f"  • {sym:<12}: Status={data.get('status_badge', data.get('status'))} | "
                      f"Roll10 Acc={data.get('rolling_10_acc_pct', 'N/A')}% | "
                      f"Test Acc={data.get('test_accuracy_pct', 'N/A')}%")
            print(f"\nRecent History ({len(state.get('history', []))} records):")
            for h in state.get("history", [])[:5]:
                print(f"  [{h.get('timestamp')}] {h.get('event_type').upper()}: {h.get('details', {}).get('message')}")
        return

    if args.monitor:
        logger.info("Executing Agent Monitor Task...")
        res = agent.monitor_models(symbols=symbols_list, threshold_pct=args.threshold)
        if args.json:
            print(json.dumps(res.to_dict(), indent=2))
        else:
            print(f"\n✅ {res.message}")
            for rec in res.recommendations:
                print(f"   👉 {rec}")
            print("\nModel Health Summary:")
            for sym, rep in res.data.get("health_reports", {}).items():
                print(f"  [{rep.get('status_badge')}] {sym} ({rep.get('name')}): "
                      f"Overall Acc: {rep.get('overall_accuracy_pct')}% | "
                      f"Rolling 10: {rep.get('rolling_10_acc_pct')}% | "
                      f"Samples: {rep.get('completed_samples')}")

    elif args.train:
        logger.info("Executing Agent Training Task...")
        res = agent.train_models(symbols=symbols_list, period=args.period)
        if args.json:
            print(json.dumps(res.to_dict(), indent=2))
        else:
            print(f"\n✅ {res.message}")
            for rec in res.recommendations:
                print(f"   👉 {rec}")
            print("\nTraining Metrics:")
            for sym, data in res.data.get("training_results", {}).items():
                if data.get("status") == "success":
                    print(f"  • {sym}: Test Acc = {data.get('test_accuracy_pct')}% | Bias = {data.get('direction')} ({data.get('probability_up_pct')}%) | Conf = {data.get('confidence')}")
                else:
                    print(f"  • {sym}: Failed ({data.get('error')})")

    elif args.auto:
        logger.info(f"Executing Autonomous Audit & Retrain (Threshold: {args.threshold}%)...")
        res = agent.auto_evaluate_and_retrain(symbols=symbols_list, threshold_pct=args.threshold)
        if args.json:
            print(json.dumps(res.to_dict(), indent=2))
        else:
            print(f"\n✅ {res.message}")
            for rec in res.recommendations:
                print(f"   👉 {rec}")
            drifted = res.data.get("retrained_symbols", [])
            if drifted:
                print(f"\n🔄 Models Auto-Retrained ({len(drifted)}): {', '.join(drifted)}")
            else:
                print("\n✨ All models are performing above target accuracy. No drift detected.")

if __name__ == "__main__":
    main()
