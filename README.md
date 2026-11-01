# Instamart Customer Intelligence & Support Copilot

## Overview

This project builds a customer intelligence and support copilot using synthetic Instamart-style data. It combines data cleaning, SQLite analytics, customer segmentation, lapse prediction, support ticket triage, policy question answering, and a refund agent.

All data is synthetic. It does not represent real customer records, actual Instamart policies, or real pricing.

## Reproduce

Run commands from the project root with the virtual environment activated.

```powershell
python generate_data.py
python scripts/clean_data.py
python scripts/validate_data.py
python scripts/create_database.py
python scripts/run_sql_queries.py
python scripts/verify_revenue.py
python scripts/customer_analytics.py
python scripts/build_lapse_dataset.py
python scripts/train_lapse_model.py
python scripts/triage_tickets.py
```

The Part D commands will be added after its implementation is complete. Run the data-generation step before the later steps when recreating the project from scratch.

## Part A — Data Foundation: Clean, Store, Query

### Data cleaning and validation

- Generated synthetic data containing 1,200 customers, 40 products, 13,084 raw order records, and 164 support tickets.
- Removed 119 exact duplicate order rows and applied the latest-record rule to retain one row per order ID, leaving 12,309 orders.
- Standardized city and gender values.
- Converted ages to numeric values, treated invalid ages as missing, and filled missing ages using city-level medians.
- Converted signup and order timestamps to datetime values.
- Converted order amounts to numeric values and filled missing amounts using product price, quantity, and discount.
- Replaced 105 negative delivery-time values with missing values. Missing delivery times for completed orders were imputed using medians by product category. Cancelled orders retain missing delivery times.
- The data-validation script reported that all 12 checkpoints passed.

### SQLite and SQL findings

Created `data/instamart.db` with customer, product, and order tables and ran seven SQL queries.

- Completed orders: 11,110.
- Completed-order revenue: INR 2,998,668.80.
- An independent pandas calculation matched the SQLite revenue total.
- Other queries examined top customers and products, customers without completed orders, above-average customer spending, ranking behavior, and cumulative daily revenue.

## Part B — Customer Segments and Lapse Prediction

### Customer segmentation

- Built an RFM dataset for 1,175 customers.
- Compared K-means clustering solutions from K=2 through K=8.
- Selected K=4 based on the tested silhouette scores; its silhouette score was approximately 0.526.
- Identified a Premium segment containing 138 customers.
- The first two PCA components explained approximately 85.2% of the variance.

### Lapse prediction

- Built a historical-feature dataset using a cutoff date of 2025-09-01.
- Defined the lapse label based on customers without completed orders during September 2025 who had completed orders before the cutoff.
- The cohort contained 1,156 customers, with 443 labelled lapsed (38.32%).
- Compared logistic regression and random forest models against a majority-class baseline.
- The selected logistic-regression model after training-set oversampling achieved test accuracy of approximately 71.6% and lapse-class F1 of approximately 0.697.
- Oversampling was applied to training data only.
- Saved the trained model as `outputs/lapse_model.joblib` and verified that the saved model reproduced its predictions.
- Generated `outputs/retention_list.csv` for the 138 Premium-segment customers, including 44 customers labelled lapsed.

**Limitation:** The Premium segment was generated using RFM data covering the wider dataset period, whereas the lapse label uses a September 2025 window. The retention list is therefore retrospective and should not be treated as a leakage-free prospective evaluation.

## Part C — Support Ticket Triage with Guardrails

### Privacy and input guardrails

- Processed 164 tickets in file order.
- Skipped 4 empty tickets and flagged 4 tickets exceeding the 400-token input limit. Neither group is sent for classification.
- Redacted 55 email addresses, 91 phone numbers, and 132 order IDs: 278 substitutions in total.
- Rechecked the redacted text and found zero remaining matches for the three targeted patterns.
- Added assertions to verify that those patterns do not appear in prompts sent to the mock model.

### Retry and output validation

- Used exponential backoff with jitter and a maximum of three retries after the initial attempt.
- In each full run, all 19 simulated transient errors were recovered.
- Validated JSON replies against the four permitted categories: `billing`, `delivery`, `technical`, and `fraud`.
- When JSON parsing fails, extracted lowercase words with a regular expression and accepted a category only when exactly one permitted category word was found.
- Invalid outputs are flagged and routed to human review rather than guessed.

### Zero-shot and few-shot comparison

Both modes classified 147 tickets successfully when JSON and keyword classifications are counted together.

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Accuracy | 85.71% | 100.00% |
| Total tokens | 8,308 | 15,796 |
| Tokens per classified ticket | 56.52 | 107.46 |

The zero-shot confusion table showed 11 fraud tickets and 10 technical tickets incorrectly predicted as billing. The few-shot run achieved 100% accuracy on this synthetic test set.

The few-shot prompt used approximately 90% more tokens per classified ticket, but improved observed accuracy by 14.29 percentage points. In this test, the extra prompt cost is worthwhile when avoiding category errors is important, especially for fraud-related tickets. These results come from a deterministic mock model and should not be interpreted as a guarantee of real-world model accuracy.

### Token budget and routing

- The zero-shot budget demonstration uses a 6,000-token budget and saves `outputs/triage_budget_demo.csv`.
- The latest verified run recorded 5,864 tokens and stopped 46 tickets because of budget exhaustion.
- The final few-shot run uses `TOKEN_BUDGET = 100000` and saves `outputs/triage_results.csv`.
- The final CSV has 164 rows and exactly these columns: `ticket_id`, `customer_id`, `status`, `category`, `segment_name`, `priority`, `route`, and `tokens_used`.
- Routing rules send empty, long, invalid-output, and budget-exhausted tickets to `human_review` with `NORMAL` priority. Valid fraud classifications go to `security_team` with `HIGH` priority. Other Premium customers go to `premium_desk` with `HIGH` priority, and remaining tickets go to `support_queue` with `NORMAL` priority.
- The automated routing check reported zero errors for the human-review and fraud-routing rules.

**Limitation:** The budget demo uses estimated prompt cost plus a 100-token reserve to decide whether to attempt a call. This is a practical estimate, not a universal guarantee for a real model with variable completion lengths. The recorded usage must always be checked against the actual budget.


## Part D — Policy RAG Assistant and Refund Agent

### Policy document retrieval

- Loaded five policy files: refund, delivery, escalation, payment, and privacy.
- Split the policies into 40-word chunks with 10-word overlap, using a 30-word step.
- Generated 15 chunks in total: refund (4), delivery (3), escalation (3), payment (3), and privacy (2).
- Used TF-IDF vectorization and cosine similarity to retrieve the three most relevant chunks.
- The query `refund policy late delivery` returned three chunks from `refund_policy.txt`.

**TF-IDF explanation:** Term Frequency (TF) measures how often a word appears in a chunk. Inverse Document Frequency (IDF) gives greater importance to words that appear in fewer chunks. Their combination helps retrieve policy chunks relevant to a question.

### RAG evaluation

Evaluated the assistant against 14 questions in `policy_eval.csv`: 12 answerable and 2 unanswerable.

| Evaluation metric | 10-word overlap | No overlap |
|---|---:|---:|
| Expected document in top 3 | 12/12 | 12/12 |
| Expected fact in answer | 12/12 | 10/12 |
| Correct fallback answers | 2/2 | 2/2 |

The 10-word overlap configuration achieved all expected results. Without overlap, two expected facts were missed, demonstrating how information split across chunk boundaries can affect retrieval and answer generation.

**Limitation:** TF-IDF relies on word matching rather than semantic understanding. Questions using different words for the same concept may retrieve less relevant chunks.

### Refund agent and tools

Implemented a ReAct-style agent using the offline `mock_llm.py` model and five tools:

- `get_order`: retrieves order details from SQLite.
- `search_policy`: retrieves relevant policy chunks.
- `calculate_refund`: calculates the eligible refund amount.
- `issue_refund`: records an eligible refund in the refund ledger.
- `escalate_to_human`: records cases requiring human approval.

The refund tools check order existence, completion status, delivery time, and refund amount. Refunds above INR 1,000 require human approval.

### Scenario evaluation

| Scenario | Expected result | Actual result | Model calls |
|---|---:|---:|---:|
| S1 | REFUND ₹128.25 | REFUND ₹128.25 | 5 |
| S2 | REFUND ₹135 | REFUND ₹135 | 5 |
| S3 | ESCALATE ₹1,999 | ESCALATE ₹1,999 | 5 |
| S4 | DENIED | DENIED | 4 |
| S5 | DENIED | DENIED | 3 |
| S6 | NOT_FOUND | NOT_FOUND | 2 |
| S7 | FALLBACK_MAX_ITERATIONS | FALLBACK_MAX_ITERATIONS | 6 |

The scenario results are saved in `outputs/agent_runs.jsonl`, and refund or escalation records are saved in `outputs/refund_ledger.jsonl`.

### Safety limits and checkpoint recovery

- A token-limit test returned `FALLBACK_TOKEN_LIMIT` when the configured limit was 200 tokens.
- A zero-second time limit returned `FALLBACK_TIMEOUT`.
- The agent saves its conversation messages, token usage, step number, and executed tools in checkpoint files under `state/`.
- Simulated a crash during S1 after two tools, then resumed the agent.
- The resumed run issued the expected INR 128.25 refund without repeating previously executed tools.

### ReAct versus plan-and-execute

ReAct was selected because it chooses the next action based on the latest tool observation. This is useful when an order is missing, cancelled, or requires human approval.

In the tested scenarios, S1 used 5 model calls and 1,034 tokens, S5 used 3 calls and 406 tokens, and S6 used 2 calls and 109 tokens. A fixed plan could be simpler for predictable cases, but it may not respond as naturally to unexpected tool results. ReAct was therefore chosen for its ability to adapt the next step to the observed result.

**Policy limitation:** The written refund policy includes a seven-day request window. Some supplied scenario timestamps fall outside that window, although the required scenario outcomes expect refunds or escalation. The seven-day eligibility checker is kept separate and is not enforced in these scripted scenario outcomes. This inconsistency must be resolved before using the agent for real refund decisions.

## Limitations and Next Steps

- The refund agent's seven-day eligibility rule is not enforced in the scripted scenario outcomes because of inconsistencies between the supplied scenario timestamps and expected results.
- Run all scripts from a clean state and verify that every required output is generated.
- A production deployment should enforce policy eligibility, prevent duplicate refund transactions, and use transactional logging and stronger authorization checks.
