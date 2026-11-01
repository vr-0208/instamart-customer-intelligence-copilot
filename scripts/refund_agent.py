
import sys
import json
import sqlite3
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add project root to Python import path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import mock_llm
from policy_rag import PolicyRAG


# Project paths
DB_PATH = BASE_DIR / "data" / "instamart.db"
OUTPUT_DIR = BASE_DIR / "outputs"
STATE_DIR = BASE_DIR / "state"

LEDGER_FILE = OUTPUT_DIR / "refund_ledger.jsonl"

OUTPUT_DIR.mkdir(exist_ok=True)
STATE_DIR.mkdir(exist_ok=True)


# Initialize Policy RAG
rag = PolicyRAG()

TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Retrieve order details using an order ID.",
        "parameters": {
            "order_id": "string"
        }
    },
    {
        "name": "search_policy",
        "description": "Search the policy documents for relevant rules.",
        "parameters": {
            "query": "string"
        }
    },
    {
        "name": "calculate_refund",
        "description": "Calculate the eligible refund based on delivery time.",
        "parameters": {
            "order_amount": "number",
            "delivery_minutes": "number"
        }
    },
    {
        "name": "issue_refund",
        "description": "Record an approved refund.",
        "parameters": {
            "order_id": "string",
            "refund_inr": "number"
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Send a refund request above INR 1000 for human approval.",
        "parameters": {
            "order_id": "string",
            "refund_inr": "number"
        }
    }
]

# TOOL 1: Get order details from SQLite database


def get_order(order_id):
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                order_id,
                status,
                order_amount,
                delivery_minutes,
                order_ts
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,)
        )

        row = cursor.fetchone()

        if row is None:
            return {"error": "order not found"}

        return {
            "order_id": row["order_id"],
            "status": row["status"],
            "order_amount": row["order_amount"],
            "delivery_minutes": row["delivery_minutes"],
            "order_ts": row["order_ts"]
        }

    finally:
        connection.close()


# Check whether an order is within the 7-day refund window
def check_seven_day_eligibility(order, reference_date=None):
    if order.get("status") != "completed":
        return False

    if order.get("delivery_minutes") is None:
        return False

    order_time = datetime.fromisoformat(order["order_ts"])

    delivery_time = order_time + timedelta(
        minutes=float(order["delivery_minutes"])
    )

    if reference_date is None:
        reference_date = datetime.now()

    days_since_delivery = (
        reference_date - delivery_time
    ).total_seconds() / 86400

    return 0 <= days_since_delivery <= 7



# TOOL 2: Search policy documents
def search_policy(query):
    results = rag.retrieve(query, k=3)

    return {
        "results": results
    }



# TOOL 3: Calculate refund amount
def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is None:
        return {"refund_inr": 0.0}

    if float(delivery_minutes) > 45:
        refund = round(float(order_amount), 2)
    else:
        refund = 0.0

    return {
        "refund_inr": refund
    }


# Helper: Record refund or escalation in JSONL ledger
def record_ledger(order_id, refund_inr, tool_name):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
        "tool_name": tool_name
    }

    with LEDGER_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


# TOOL 4: Issue refund

# TOOL 4: Issue refund
def issue_refund(order_id, refund_inr):
    refund_inr = round(float(refund_inr), 2)

    if refund_inr <= 0 or refund_inr > 1000:
        return {"error": "refund amount not eligible for automatic issue"}

    order = get_order(order_id)

    if "error" in order:
        return {"error": "order not found"}

    if order["status"] != "completed":
        return {"error": "only completed orders qualify"}

    if order["delivery_minutes"] is None or order["delivery_minutes"] <= 45:
        return {"error": "delivery does not qualify for a refund"}

    record_ledger(order_id, refund_inr, "issue_refund")

    return {"refund_inr": refund_inr}


# TOOL 5: Escalate refund to human

# TOOL 5: Escalate refund to human
def escalate_to_human(order_id, refund_inr):
    refund_inr = round(float(refund_inr), 2)

    if refund_inr <= 1000:
        return {"error": "human escalation is only for refunds above INR 1000"}

    order = get_order(order_id)

    if "error" in order:
        return {"error": "order not found"}

    if order["status"] != "completed":
        return {"error": "only completed orders qualify"}

    if order["delivery_minutes"] is None or order["delivery_minutes"] <= 45:
        return {"error": "delivery does not qualify for a refund"}

    if refund_inr != round(float(order["order_amount"]), 2):
        return {"error": "refund amount does not match order amount"}

    record_ledger(order_id, refund_inr, "escalate_to_human")

    return {
        "refund_inr": refund_inr,
        "status": "pending_human_approval"
    }


# Connect tool names to Python functions
TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_tools=None
):
    start_time = time.monotonic()

    messages = [
        {
            "role": "system",
            "content": (
                "AGENT: You are an Instamart refund support agent. "
                "Follow the refund policy. Only completed orders qualify. "
                "Refunds above INR 1000 require human approval."
            )
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    steps = []
    tokens_used = 0
    llm_calls = 0

    status = None
    final_answer = None

    for iteration in range(max_iterations):

        # Time limit check
        if time.monotonic() - start_time >= time_limit_seconds:
            status = "FALLBACK_TIMEOUT"
            final_answer = "Request sent to human support due to time limit."
            break

        # Call mock LLM
        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1

        tokens_used += response.usage.total_tokens

        # Token limit check
        if tokens_used > token_limit:
            status = "FALLBACK_TOKEN_LIMIT"
            final_answer = "Request sent to human support due to token limit."
            break

        assistant_message = response.choices[0].message

        # No tool call means the model has finished
        if not assistant_message.tool_calls:
            final_answer = assistant_message.content or ""
            status = final_answer.split(":")[0].strip()
            break

        # Add assistant message containing tool calls
        messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments)
                    }
                }
                for call in assistant_message.tool_calls
            ]
        })

        # Execute requested tools
        for call in assistant_message.tool_calls:

            tool_name = call.name
            arguments = call.arguments

            if tool_name not in TOOL_MAP:
                observation = {"error": "unknown tool"}
            else:
                observation = TOOL_MAP[tool_name](**arguments)

            steps.append({
                "tool": tool_name,
                "arguments": arguments,
                "observation": observation
            })

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": tool_name,
                "content": json.dumps(observation, default=str)
            })

        # Save checkpoint after tool observations
        checkpoint = {
    "run_id": scenario_id,
    "scenario_id": scenario_id,
    "messages": messages,
    "tokens_used": tokens_used,
    "llm_calls": llm_calls,
    "steps": steps,
    "step_number": len(steps),
    "tools_executed": [
        step["tool"] for step in steps
    ]
}

        checkpoint_path = STATE_DIR / f"{scenario_id}.json"

        checkpoint_path.write_text(
            json.dumps(checkpoint, indent=2, default=str),
            encoding="utf-8"
        )
                # Simulate a crash only after the checkpoint is safely saved
        if crash_after_tools is not None and len(steps) >= crash_after_tools:
            raise RuntimeError("SIMULATED_CRASH")

    else:
        status = "FALLBACK_MAX_ITERATIONS"
        final_answer = "Request sent to human support for further review."

    if status is None:
        status = "FALLBACK_MAX_ITERATIONS"
        final_answer = "Request sent to human support for further review."

    return {
        "status": status,
        "final_answer": final_answer,
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used
    }

def resume_agent(run_id, max_iterations=6,
                 token_limit=6000, time_limit_seconds=10):

    checkpoint_path = STATE_DIR / f"{run_id}.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {run_id}"
        )

    checkpoint = json.loads(
        checkpoint_path.read_text(encoding="utf-8")
    )

    messages = checkpoint["messages"]
    steps = checkpoint.get("steps", [])
    tokens_used = checkpoint["tokens_used"]
    llm_calls = checkpoint.get("llm_calls", 0)

    start_time = time.monotonic()

    status = None
    final_answer = None

    for iteration in range(llm_calls, max_iterations):

        if time.monotonic() - start_time >= time_limit_seconds:
            status = "FALLBACK_TIMEOUT"
            final_answer = "Request sent to human support due to time limit."
            break

        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += response.usage.total_tokens

        if tokens_used > token_limit:
            status = "FALLBACK_TOKEN_LIMIT"
            final_answer = "Request sent to human support due to token limit."
            break

        assistant_message = response.choices[0].message

        if not assistant_message.tool_calls:
            final_answer = assistant_message.content or ""
            status = final_answer.split(":")[0].strip()
            break

        messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments)
                    }
                }
                for call in assistant_message.tool_calls
            ]
        })

        for call in assistant_message.tool_calls:
            tool_name = call.name
            arguments = call.arguments

            if tool_name not in TOOL_MAP:
                observation = {"error": "unknown tool"}
            else:
                observation = TOOL_MAP[tool_name](**arguments)

            steps.append({
                "tool": tool_name,
                "arguments": arguments,
                "observation": observation
            })

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": tool_name,
                "content": json.dumps(observation, default=str)
            })

        updated_checkpoint = {
            "run_id": run_id,
            "scenario_id": checkpoint["scenario_id"],
            "messages": messages,
            "tokens_used": tokens_used,
            "llm_calls": llm_calls,
            "steps": steps,
            "step_number": len(steps),
            "tools_executed": [
                step["tool"] for step in steps
            ]
        }

        checkpoint_path.write_text(
            json.dumps(updated_checkpoint, indent=2, default=str),
            encoding="utf-8"
        )

    else:
        status = "FALLBACK_MAX_ITERATIONS"
        final_answer = "Request sent to human support for further review."

    if status is None:
        status = "FALLBACK_MAX_ITERATIONS"
        final_answer = "Request sent to human support for further review."

    return {
        "status": status,
        "final_answer": final_answer,
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used
    }

