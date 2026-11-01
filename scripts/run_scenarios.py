
import csv
import json
import sys
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from refund_agent import run_agent

SCENARIO_FILE = BASE_DIR / "data" / "raw" / "agent_scenarios.csv"
OUTPUT_FILE = BASE_DIR / "outputs" / "agent_runs.jsonl"


def main():
    results = []

    with SCENARIO_FILE.open(
        "r", encoding="utf-8", newline=""
    ) as file:
        scenarios = list(csv.DictReader(file))

    with OUTPUT_FILE.open("w", encoding="utf-8") as output:

        for scenario in scenarios:
            scenario_id = scenario["scenario_id"]
            message = scenario["message"]

            print(f"\nRunning {scenario_id}...")

            result = run_agent(
                user_message=message,
                scenario_id=scenario_id,
                max_iterations=6,
                token_limit=6000,
                time_limit_seconds=10
            )

            # Extract refund amount from the final tool observation
            refund_amount = 0.0

            for step in result["steps"]:
                if step["tool"] in (
                    "issue_refund",
                    "escalate_to_human"
                ):
                    refund_amount = step["observation"].get(
                        "refund_inr", 0.0
                    )

            record = {
                "scenario_id": scenario_id,
                "status": result["status"],
                "final_answer": result["final_answer"],
                "refund_inr": round(float(refund_amount), 2),
                "llm_calls": result["llm_calls"],
                "tokens_used": result["tokens_used"],
                "steps": result["steps"]
            }

            output.write(json.dumps(record) + "\n")
            results.append(record)

            print("Status:", result["status"])
            print("Refund:", record["refund_inr"])
            print("Model calls:", result["llm_calls"])
            print("Tokens:", result["tokens_used"])

    print("\n" + "=" * 45)
    print("ALL SCENARIOS COMPLETED")
    print("=" * 45)

    print("\nScenario Summary:")

    for result in results:
        print(
            result["scenario_id"],
            "|",
            result["status"],
            "| INR",
            result["refund_inr"],
            "| Calls:",
            result["llm_calls"]
        )

    print("\nSaved results to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
