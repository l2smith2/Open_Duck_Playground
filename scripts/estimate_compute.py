"""Choose Kaggle or paid compute from the measured 20M-step runtime."""

import argparse
import json
from pathlib import Path

FULL_TO_BENCHMARK_RATIO = 15
KAGGLE_SESSION_LIMIT_HOURS = 12.0
KAGGLE_SAFETY_LIMIT_HOURS = 10.0
MAX_PAID_RATE_USD_PER_HOUR = 0.50
PAID_STOP_BUDGET_USD = 8.0
PAID_TOTAL_CAP_USD = 10.0


def estimate(benchmark_seconds: float, paid_rate: float) -> dict:
    if benchmark_seconds <= 0:
        raise ValueError("benchmark_seconds must be positive")
    if paid_rate <= 0 or paid_rate > MAX_PAID_RATE_USD_PER_HOUR:
        raise ValueError("paid_rate must be greater than 0 and no more than US$0.50/hour")
    projected_hours = benchmark_seconds * FULL_TO_BENCHMARK_RATIO / 3600
    paid_cost = projected_hours * paid_rate
    return {
        "benchmark_seconds": benchmark_seconds,
        "projected_300m_hours": projected_hours,
        "kaggle_session_limit_hours": KAGGLE_SESSION_LIMIT_HOURS,
        "kaggle_safety_limit_hours": KAGGLE_SAFETY_LIMIT_HOURS,
        "recommendation": "kaggle" if projected_hours < KAGGLE_SAFETY_LIMIT_HOURS else "runpod",
        "projected_paid_cost_usd": paid_cost,
        "paid_rate_usd_per_hour": paid_rate,
        "paid_stop_budget_usd": PAID_STOP_BUDGET_USD,
        "paid_total_cap_usd": PAID_TOTAL_CAP_USD,
        "within_paid_stop_budget": paid_cost <= PAID_STOP_BUDGET_USD,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-seconds", type=float, required=True)
    parser.add_argument("--paid-rate", type=float, default=0.50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = estimate(args.benchmark_seconds, args.paid_rate)
    output = json.dumps(result, indent=2) + "\n"
    print(output, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    if result["recommendation"] == "runpod" and not result["within_paid_stop_budget"]:
        raise SystemExit(
            "Projected paid work exceeds the US$8 stop budget. Split the run or reduce scope."
        )


if __name__ == "__main__":
    main()
