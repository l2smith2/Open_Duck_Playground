"""Fail closed before a paid run can exceed the US$8 working budget."""

import argparse
import json

MAX_RATE = 0.50
STOP_BUDGET = 8.0
TOTAL_CAP = 10.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--elapsed-hours", type=float, default=0.0)
    parser.add_argument("--planned-hours", type=float, required=True)
    args = parser.parse_args()
    if not 0 < args.rate <= MAX_RATE:
        raise SystemExit("Use a Community RTX 4090 priced at no more than US$0.50/hour")
    projected = (args.elapsed_hours + args.planned_hours) * args.rate
    result = {
        "projected_working_cost_usd": projected,
        "stop_budget_usd": STOP_BUDGET,
        "reserved_recovery_budget_usd": TOTAL_CAP - STOP_BUDGET,
        "allowed": projected <= STOP_BUDGET,
    }
    print(json.dumps(result, indent=2))
    if not result["allowed"]:
        raise SystemExit("Paid run refused: projected working cost exceeds US$8")


if __name__ == "__main__":
    main()
