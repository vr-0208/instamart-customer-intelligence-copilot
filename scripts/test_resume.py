
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from refund_agent import run_agent, resume_agent

message = (
    "My order ORD-100048 reached me very late. "
    "I would like a refund."
)

# Step 1: Simulate a crash after two tools
try:
    run_agent(
        message,
        "CRASH_S1",
        crash_after_tools=2
    )
except RuntimeError as error:
    print("Crash:", error)

# Step 2: Resume from saved checkpoint
result = resume_agent("CRASH_S1")

print("\nRESUME TEST RESULTS")
print("=" * 40)
print("Final status:", result["status"])
print("Answer:", result["final_answer"])
print("Model calls:", result["llm_calls"])
print("Tools:", [step["tool"] for step in result["steps"]])
print("Final observation:", result["steps"][-1]["observation"])
