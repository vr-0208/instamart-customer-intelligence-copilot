
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mock_llm

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MAX_INPUT_TOKENS = 400


def make_messages(text, few_shot=False):
    if few_shot:
        system_prompt = """
CLASSIFY
Classify the ticket into billing, delivery, technical, or fraud.
Return JSON only: {"category": "billing"}.
EXAMPLES:
Ticket: I was charged twice.
Category: billing
Ticket: My delivery is late.
Category: delivery
Ticket: The app crashes.
Category: technical
Ticket: My account was hacked.
Category: fraud
"""
    else:
        system_prompt = """
CLASSIFY
Classify the ticket into billing, delivery, technical, or fraud.
Return JSON only: {"category": "billing"}.
"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tickets = pd.read_csv(
        OUTPUT_DIR / "tickets_redacted.csv",
        keep_default_na=False,
    )

    eligible = tickets[
        (tickets["triage_status"] == "ready")
        & (tickets["estimated_tokens"] <= MAX_INPUT_TOKENS)
    ]

    rows = []

    for _, ticket in eligible.iterrows():
        for mode, few_shot in [
            ("zero_shot", False),
            ("few_shot", True),
        ]:
            text = ticket["redacted_text"]
            messages = make_messages(text, few_shot)

            input_tokens = sum(
                mock_llm.est_tokens(message["content"])
                for message in messages
            )

            if input_tokens > MAX_INPUT_TOKENS:
                rows.append({
                    "ticket_id": ticket["ticket_id"],
                    "mode": mode,
                    "status": "skipped_budget",
                    "prompt_tokens": input_tokens,
                    "completion_tokens": 0,
                    "total_tokens": input_tokens,
                })
                continue

            mock_llm.reset()

            try:
                response = mock_llm.chat(messages)
                usage = response.usage

                rows.append({
                    "ticket_id": ticket["ticket_id"],
                    "mode": mode,
                    "status": "measured",
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                })

            except mock_llm.TransientLLMError:
                rows.append({
                    "ticket_id": ticket["ticket_id"],
                    "mode": mode,
                    "status": "transient_error",
                    "prompt_tokens": input_tokens,
                    "completion_tokens": 0,
                    "total_tokens": input_tokens,
                })

    result = pd.DataFrame(rows)
    output_path = OUTPUT_DIR / "triage_budget_demo.csv"
    result.to_csv(output_path, index=False)

    print("TOKEN BUDGET DEMONSTRATION")
    print("=" * 40)
    print("Eligible tickets:", len(eligible))
    print("Measurement rows:", len(result))
    print("\nStatus counts:")
    print(result["status"].value_counts().to_dict())
    print("\nUsage totals by prompt mode:")
    print(
        result.groupby("mode")[
            ["prompt_tokens", "completion_tokens", "total_tokens"]
        ].sum().to_string()
    )
    print("\nSaved to:", output_path)


if __name__ == "__main__":
    main()
