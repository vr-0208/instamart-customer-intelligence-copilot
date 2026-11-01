
from pathlib import Path
import json
import random
import re
import sys
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mock_llm

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_PATTERN = re.compile(r"\b[6-9]\d{9}\b")
ORDER_ID_PATTERN = re.compile(r"ORD-\d{6}")

ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
TOKEN_BUDGET = 100000


def redact_text(text):
    text = EMAIL_PATTERN.sub("[EMAIL]", text)
    text = PHONE_PATTERN.sub("[PHONE]", text)
    text = ORDER_ID_PATTERN.sub("[ORDER_ID]", text)
    return text


def make_messages(text, few_shot=False):
    system_prompt = """
CLASSIFY
Choose exactly one category: billing, delivery, technical, fraud.
Return JSON only in this format:
{"category": "...", "reason": "..."}
"""

    if few_shot:
        system_prompt += """
EXAMPLES:
Ticket: I was charged twice.
Answer: {"category": "billing", "reason": "Duplicate charge"}
Ticket: My delivery is late.
Answer: {"category": "delivery", "reason": "Late delivery"}
Ticket: Someone used my account without permission.
Answer: {"category": "fraud", "reason": "Unauthorized account use"}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    # Prove that sensitive identifiers never enter a prompt.
    prompt_text = "\n".join(message["content"] for message in messages)
    assert not EMAIL_PATTERN.search(prompt_text)
    assert not PHONE_PATTERN.search(prompt_text)
    assert not ORDER_ID_PATTERN.search(prompt_text)
    assert messages[-1]["content"] == text

    return messages


def classify_ticket(ticket_id, text, few_shot=False):
    messages = make_messages(text, few_shot)

    for attempt in range(4):  # initial call + maximum 3 retries
        try:
            response = mock_llm.chat(messages)
            content = response.choices[0].message.content
            usage = response.usage

            try:
                result = json.loads(content)
                category = result.get("category")

                if category in ALLOWED_CATEGORIES:
                    status = "classified_json"
                else:
                    category = ""
                    status = "flagged_invalid_output"

            except (json.JSONDecodeError, TypeError, AttributeError):
                words = re.findall(r"[a-z]+", content.lower())
                found = [word for word in words if word in ALLOWED_CATEGORIES]

                if len(found) == 1:
                    category = found[0]
                    status = "classified_keyword"
                else:
                    category = ""
                    status = "flagged_invalid_output"

            return {
                "category": category,
                "status": status,
                "tokens_used": int(usage.total_tokens),
                "attempts": attempt + 1,
            }

        except mock_llm.TransientLLMError:
            if attempt == 3:
                raise

            wait = BASE_DELAY * (2 ** attempt) * random.uniform(0.9, 1.15)
            print(
                f"Retry: ticket={ticket_id}, "
                f"retry_number={attempt}, wait={wait:.4f}s"
            )
            time.sleep(wait)

    raise RuntimeError(f"Retries exhausted for {ticket_id}")


def prepare_tickets():
    tickets = pd.read_csv(RAW_DIR / "tickets.csv", keep_default_na=False)

    tickets["original_tokens"] = tickets["text"].apply(mock_llm.est_tokens)
    tickets["redacted_text"] = tickets["text"].apply(redact_text)

    tickets["status"] = "ready"
    empty = tickets["text"].str.strip().eq("")
    long = tickets["original_tokens"] > MAX_INPUT_TOKENS

    tickets.loc[empty, "status"] = "skipped_empty"
    tickets.loc[long & ~empty, "status"] = "flagged_long"

    return tickets


def print_privacy_audit(tickets):
    counts = {
        "emails": sum(len(EMAIL_PATTERN.findall(t)) for t in tickets["text"]),
        "phones": sum(len(PHONE_PATTERN.findall(t)) for t in tickets["text"]),
        "order_ids": sum(len(ORDER_ID_PATTERN.findall(t)) for t in tickets["text"]),
    }

    remaining = {
        "emails": sum(len(EMAIL_PATTERN.findall(t)) for t in tickets["redacted_text"]),
        "phones": sum(len(PHONE_PATTERN.findall(t)) for t in tickets["redacted_text"]),
        "order_ids": sum(len(ORDER_ID_PATTERN.findall(t)) for t in tickets["redacted_text"]),
    }

    print("\nPRIVACY AUDIT")
    print("Substitutions:", counts)
    print("Total substitutions:", sum(counts.values()))
    print("Remaining identifiers:", remaining)

    assert remaining == {"emails": 0, "phones": 0, "order_ids": 0}


def run_comparison(tickets, few_shot):
    mock_llm.reset()
    results = []
    running_tokens = 0

    for _, ticket in tickets.iterrows():
        status = ticket["status"]
        category = ""
        tokens_used = 0

        if status == "ready":
            result = classify_ticket(
                ticket["ticket_id"],
                ticket["redacted_text"],
                few_shot=few_shot,
            )
            category = result["category"]
            status = result["status"]
            tokens_used = result["tokens_used"]
            running_tokens += tokens_used

        results.append({
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "true_category": ticket["true_category"],
            "status": status,
            "category": category,
            "tokens_used": tokens_used,
        })

    result_df = pd.DataFrame(results)
    classified = result_df["status"].isin(
        ["classified_json", "classified_keyword"]
    )
    accuracy = (
        result_df.loc[classified, "category"]
        == result_df.loc[classified, "true_category"]
    ).mean()

    print("\n", "FEW-SHOT" if few_shot else "ZERO-SHOT", "COMPARISON")
    print("Status counts:", result_df["status"].value_counts().to_dict())
    print("Classified:", int(classified.sum()))
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Total tokens:", running_tokens)
    print("Tokens per classified ticket:",
          round(running_tokens / max(int(classified.sum()), 1), 2))

    if not few_shot:
        print("\nZERO-SHOT CONFUSION TABLE")
        print(pd.crosstab(
            result_df.loc[classified, "true_category"],
            result_df.loc[classified, "category"],
            rownames=["Actual"],
            colnames=["Predicted"],
        ))

    return result_df



def run_budget_demo(tickets):
    mock_llm.reset()
    budget = 6000
    running_tokens = 0
    rows = []

    for _, ticket in tickets.iterrows():
        status = ticket["status"]
        category = ""
        tokens_used = 0

        if status == "ready":
            messages = make_messages(
                ticket["redacted_text"],
                few_shot=False,
            )

            # Estimate the next call's token cost before making the call.
            estimated_prompt_tokens = sum(
                mock_llm.est_tokens(message["content"])
                for message in messages
            )

# Reserve extra capacity for the model's completion tokens.
            estimated_call_tokens = estimated_prompt_tokens + 100
            if running_tokens + estimated_call_tokens > budget:
                status = "budget_exhausted"
            else:
                result = classify_ticket(
                    ticket["ticket_id"],
                    ticket["redacted_text"],
                    few_shot=False,
                )

                category = result["category"]
                status = result["status"]
                tokens_used = result["tokens_used"]

                if running_tokens + tokens_used > budget:
                    # Record the call, but flag that the estimate was insufficient.
                    status = "budget_exhausted"
                    category = ""

                running_tokens += tokens_used

        rows.append({
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "status": status,
            "category": category,
            "tokens_used": tokens_used,
            "running_tokens": running_tokens,
        })

    output = OUTPUT_DIR / "triage_budget_demo.csv"
    pd.DataFrame(rows).to_csv(output, index=False)

    exhausted = sum(
        row["status"] == "budget_exhausted"
        for row in rows
    )

    print("\nBUDGET DEMO")
    print("Budget:", budget)
    print("Tokens used:", running_tokens)
    print("Tickets stopped by budget:", exhausted)
    print("Saved:", output)

    if running_tokens > budget:
        print(
            "WARNING: Actual token usage exceeded the budget. "
            "The mock response usage was higher than estimated."
        )

    if exhausted == 0:
        raise RuntimeError("Budget demo did not exhaust its budget.")



def run_final(tickets):
    mock_llm.reset()
    segments = pd.read_csv(OUTPUT_DIR / "customer_segments.csv")
    segment_lookup = segments.set_index("customer_id").to_dict("index")

    running_tokens = 0
    rows = []

    for _, ticket in tickets.iterrows():
        ticket_id = ticket["ticket_id"]
        customer_id = ticket["customer_id"]
        status = ticket["status"]
        category = ""
        tokens_used = 0

        if status == "ready":
            if running_tokens >= TOKEN_BUDGET:
                status = "budget_exhausted"
            else:
                result = classify_ticket(
                    ticket_id,
                    ticket["redacted_text"],
                    few_shot=True,
                )
                category = result["category"]
                status = result["status"]
                tokens_used = result["tokens_used"]
                running_tokens += tokens_used

        segment = segment_lookup.get(customer_id, {})
        segment_name = segment.get("segment_name", "")
        is_premium = int(segment.get("is_premium", 0) or 0) == 1

        if status in {
            "skipped_empty", "flagged_long",
            "flagged_invalid_output", "budget_exhausted",
        }:
            route, priority = "human_review", "NORMAL"
        elif category == "fraud":
            route, priority = "security_team", "HIGH"
        elif is_premium:
            route, priority = "premium_desk", "HIGH"
        else:
            route, priority = "support_queue", "NORMAL"

        rows.append({
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": status,
            "category": category,
            "segment_name": segment_name,
            "priority": priority,
            "route": route,
            "tokens_used": tokens_used,
        })

    output = OUTPUT_DIR / "triage_results.csv"
    result_df = pd.DataFrame(rows, columns=[
        "ticket_id", "customer_id", "status", "category",
        "segment_name", "priority", "route", "tokens_used",
    ])
    result_df.to_csv(output, index=False)

    print("\nFINAL FEW-SHOT RUN")
    print("Status counts:", result_df["status"].value_counts().to_dict())
    print("Total tokens:", running_tokens)
    print("Routing counts:", result_df["route"].value_counts().to_dict())
    print("Saved:", output)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tickets = prepare_tickets()

    print("INPUT GUARDRAILS")
    print("Ticket count:", len(tickets))
    print("Status counts:", tickets["status"].value_counts().to_dict())
    print("Long tickets:", int((tickets["status"] == "flagged_long").sum()))
    print("Empty tickets:", int((tickets["status"] == "skipped_empty").sum()))

    print_privacy_audit(tickets)

    redacted_path = OUTPUT_DIR / "tickets_redacted.csv"
    tickets.to_csv(redacted_path, index=False)
    print("Saved redacted tickets:", redacted_path)

    run_comparison(tickets, few_shot=False)
    run_comparison(tickets, few_shot=True)
    run_budget_demo(tickets)
    run_final(tickets)


if __name__ == "__main__":
    main()
