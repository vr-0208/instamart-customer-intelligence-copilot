# Instamart Customer Intelligence & Support Copilot

## Overview

This project builds a customer intelligence and support copilot using synthetic Instamart-style data. It combines data cleaning, SQLite analytics, customer segmentation, lapse prediction, support ticket triage, policy question answering, and a refund agent.

All data is synthetic. It does not represent real customer records, actual Instamart policies, or real pricing.

## Reproduce

Run all commands from the project root. The following instructions use Windows PowerShell.

### 1. Create and activate the virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Generate and clean the data

```powershell
python generate_data.py
python scripts/clean_data.py
python scripts/validate_data.py
```

### 4. Create the database and run SQL analysis

```powershell
python scripts/create_database.py
python scripts/run_sql_queries.py
python scripts/verify_revenue.py
```

### 5. Run customer analytics and lapse prediction

```powershell
python scripts/customer_analytics.py
python scripts/build_lapse_dataset.py
python scripts/train_lapse_model.py
```

### 6. Run support ticket triage

```powershell
python scripts/triage_tickets.py
```

### 7. Run the policy RAG assistant and refund agent

```powershell
python scripts/policy_rag.py
python scripts/run_scenarios.py
python scripts/test_resume.py
```

The scripts generate the cleaned datasets, SQLite database, trained model, evaluation results, agent logs, refund ledger, and checkpoint files in their respective directories.
## Part A 

### 1. Raw Data Audit

The project generated three synthetic datasets containing 1,200 customers, 40 products, and 13,084 raw order records.

| Dataset | Rows | Columns | Important missing values |
|---|---:|---:|---|
| Customers | 1,200 | 5 | Age: 61 (5.08%); Gender: 30 (2.50%) |
| Products | 40 | 4 | None |
| Orders | 13,084 | 9 | Delivery minutes: 2,541 (19.42%) |

Additional raw-data quality findings:

| Data-quality check | Result |
|---|---:|
| Exact duplicate order rows | 119 |
| Distinct order IDs | 12,309 |
| Pending orders | 656 |
| Nonnumeric order amounts | 120 |
| Negative delivery times | 106 |
| Blank delivery times | 2,541 |
| Customer ages outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Nonblank gender spellings | 10 |

The raw city and gender values contained inconsistent capitalization and formatting. Customer age and gender fields also contained missing values.

### 2. Data Cleaning and Validation

The cleaning process applied the following operations:

- Removed 119 exact duplicate order rows.
- Applied the latest-record rule to retain one record per order ID, resulting in 12,309 orders.
- Standardized city names and gender values.
- Converted customer ages to numeric values and handled invalid or missing ages using city-level median imputation.
- Converted signup dates and order timestamps into datetime values.
- Converted order amounts into numeric values and filled missing amounts using product price, quantity, and discount information.
- Replaced 105 negative delivery-time values with missing values.
- Imputed missing delivery times for completed orders using product-category medians.
- Retained missing delivery times for cancelled orders because those orders were not completed.

The cleaned datasets contain:

| Dataset | Cleaned records |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Orders | 12,309 |

Cleaned customer statistics:

City distribution

| City | Customers |
|---|---:|
| Bengaluru | 197 |
| Chennai | 191 |
| Delhi | 210 |
| Hyderabad | 191 |
| Mumbai | 184 |
| Pune | 227 |

Gender distribution

| Gender | Customers |
|---|---:|
| Female | 610 |
| Male | 560 |
| Unknown | 30 |

Customer ages range from 18 to 60 years. All cleaned customer records have complete age, gender, city, and signup-date fields.

The validation script completed all 12 checks successfully, including unique IDs, missing-value checks, nonnegative delivery times, completed-order delivery times, and the removal of pending orders.

### 3. SQLite Database and SQL Analysis

Created `data/instamart.db` containing three tables:

- `customers`
- `products`
- `orders`

Implemented seven SQL queries in `sql/queries.sql`.

| Query | Analysis performed | Result |
|---|---|---|
| Q1 | Display the first 30 customers | 30 customer records |
| Q2 | Completed-order revenue by city and product category | 30 city-category combinations |
| Q3 | Products priced above average catalogue price | 11 products |
| Q4 | Customers without completed orders | 25 customers |
| Q5 | Customers spending above average | 550 customers |
| Q6 | Rank customers by completed-order spending within each city | 8 differences between RANK and ROW_NUMBER |
| Q7 | Daily revenue and cumulative revenue | 183 daily records |

Q6 uses both `RANK()` and `ROW_NUMBER()` window functions. `RANK()` preserves equal spending ranks, while `ROW_NUMBER()` assigns a unique sequence using customer ID as a tie-breaker.

The five highest customer spending values observed in Q6 were:

| Customer ID | City | Completed-order spending (INR) |
|---|---|---:|
| C0082 | Bengaluru | 12,856.20 |
| C0525 | Bengaluru | 12,719.70 |
| C0699 | Bengaluru | 12,471.80 |
| C0543 | Bengaluru | 11,195.10 |
| C1147 | Bengaluru | 10,156.90 |

The daily revenue query generated 183 records, and the final cumulative revenue matched the completed-order revenue total.

### 4. Pandas Analysis

Completed-order summary:

| Metric | Value |
|---|---:|
| Completed orders | 11,110 |
| Mean order amount | INR 269.91 |
| Median order amount | INR 135.00 |
| Standard deviation |  — |
| Skewness | 4.61 |
| First quartile (Q1) | INR 64.80 |
| Third quartile (Q3) | INR 297.00 |
| Lower outlier fence | INR -283.50 |
| Upper outlier fence | INR 645.30 |
| Lower outliers | 0 |
| Upper outliers | 897 |

The completed-order amount distribution is strongly right-skewed. The mean is substantially higher than the median because a relatively small number of high-value orders increase the average.

Revenue by city and product category

| City | Dairy | Electronics | Grocery | Personal Care | Snacks | Total (INR) |
|---|---:|---:|---:|---:|---:|---:|
| Bengaluru | 87,075.05 | 233,909.45 | 110,915.65 | 20,193.85 | 71,549.40 | 523,643.40 |
| Chennai | 78,301.80 | 195,508.80 | 93,817.45 | 27,675.40 | 68,325.70 | 463,629.15 |
| Delhi | 117,271.70 | 158,549.10 | 112,658.05 | 23,471.50 | 90,879.45 | 502,829.80 |
| Hyderabad | 83,731.00 | 228,913.10 | 109,475.40 | 24,800.05 | 81,670.35 | 528,589.90 |
| Mumbai | 75,209.85 | 214,738.40 | 78,970.15 | 38,018.00 | 72,122.10 | 479,058.50 |
| Pune | 90,349.55 | 187,629.25 | 108,456.60 | 34,275.15 | 80,207.50 | 500,918.05 |
| Grand Total| 531,938.95 | 1,219,248.10| 614,293.30|168,433.95| 464,754.50 | 2,998,668.80 |

Electronics generated the highest category revenue, while Personal Care generated the lowest.

City-wise completed-order spending

| City | Revenue (INR) | Average order amount (INR) | Completed orders |
|---|---:|---:|---:|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

The correlation between order amount and discount percentage was approximately -0.095, indicating a weak negative linear relationship. Correlation does not establish causation.

An independent pandas calculation verified the SQLite completed-order revenue:

INR 2,998,668.80

### 5. Business Questions and Findings

1. Which city generates the highest revenue? 
   Hyderabad generated the highest completed-order revenue at INR 528,589.90.

2. Which city has the highest average order value? 
   Mumbai had the highest average completed-order amount at INR 297.00.

3. Is customer spending evenly distributed? 
   No. The mean order amount is INR 269.91, while the median is INR 135.00. The skewness of 4.61 indicates a strongly right-skewed distribution, with high-value orders pulling the mean upward.

4. Does discount percentage have a strong relationship with order amount?
   No strong linear relationship was observed. The correlation was approximately -0.095, indicating a weak negative association. This does not establish a causal relationship.

## Part B 

### 1. RFM Analysis and Customer Segmentation

Built an RFM (Recency, Frequency, Monetary) dataset using completed-order history for 1,175 customers.

The RFM metrics represent:

- Recency: Days since the customer's most recent completed order.
- Frequency:Number of completed orders.
- Monetary: Total completed-order spending.

RFM summary:

| Metric | Result |
|---|---:|
| Customers analysed | 1,175 |
| Median recency | 19 days |
| Minimum recency | 0 days |
| Maximum recency | 180 days |
| Average frequency | 9.46 orders |
| Maximum frequency | 33 orders |
| Average monetary value | INR 2,552.06 |
| Maximum monetary value | INR 14,766.55 |

### 2. K-Means Model Selection

Evaluated K-means clustering solutions for K values between 2 and 8 using inertia and silhouette scores.

| K | Inertia | Silhouette score |
|---|---:|---:|
| 2 | 1,820.76 | 0.465 |
| 3 | 1,268.23 | 0.445 |
| 4 | 743.07 | 0.526 |
| 5 | 617.37 | 0.480 |
| 6 | 503.17 | 0.482 |
| 7 | 408.46 | 0.416 |
| 8 | 366.50 | 0.412 |

Selected K=4 because it achieved the highest silhouette score of approximately 0.526 among the tested configurations.

Although inertia decreases as the number of clusters increases, a lower inertia alone does not establish that a clustering solution is better. The silhouette score helped evaluate cluster separation and cohesion.

### 3. Customer Segment Profiles

The selected K=4 model identified four customer segments.

| Segment | Customers | Average recency (days) | Average frequency | Average monetary value (INR) |
|---|---:|---:|---:|---:|
| Segment 0 | 436 | 28.86 | 3.77 | 1,057.62 |
| Premium | 138 | 24.12 | 6.77 | 7,129.70 |
| Segment 2 | 419 | 8.77 | 19.40 | 3,410.02 |
| Segment 3 | 182 | 109.04 | 2.20 | 685.99 |

Business interpretation and suggested actions:

- Premium: Highest average monetary value. Focus on loyalty benefits and customer retention.
- Segment 2: High-frequency customers with recent activity. Encourage repeat purchases and engagement.
- Segment 0: Lower average spending and purchase frequency. Use relevant promotions to encourage additional purchases.
- Segment 3: High recency and low purchase frequency. Consider targeted re-engagement campaigns.

The numerical cluster labels are arbitrary identifiers generated by K-means. They do not represent an inherent ranking.

### 4. PCA Analysis

Applied Principal Component Analysis to visualize the customer segments in two dimensions.

| Component | Explained variance |
|---|---:|
| Principal Component 1 | 63.4% |
| Principal Component 2 | 21.8% |
| Combined | 85.2% |

The first two principal components explain approximately 85.2% of the variance in the transformed RFM data.

### 5. Lapse Prediction Dataset

Created a historical-feature dataset using a cutoff date of 2025-09-01.

The lapse label identifies customers who had completed orders before the cutoff but did not place completed orders during September 2025.

| Metric | Result |
|---|---:|
| Customer cohort | 1,156 |
| Lapsed customers | 443 |
| Non-lapsed customers | 713 |
| Lapse percentage | 38.32% |
| Training records | 924 |
| Testing records | 232 |

Features included recency, frequency, monetary value, average discount, cancellation rate, average delivery time, tenure, age, city, gender and favourite product category.

The target label, customer ID and post-cutoff information were excluded from the model features.

### 6. Model Training and Comparison

Compared Logistic Regression and Random Forest against a majority-class baseline.

Used a preprocessing pipeline with numerical scaling and one-hot encoding for categorical variables. The train/test split was performed before model tuning, and GridSearchCV used five-fold cross-validation on training data.

Hyperparameter search

- Logistic Regression: C values of 0.01, 0.1, 1, 10 and 100.
- Random Forest: maximum depths of 3, 5 and 8, with minimum leaf sizes of 5, 10 and 20.

Best model parameters

| Model | Best parameters | Cross-validation F1 |
|---|---|---:|
| Logistic Regression | C = 100 | 0.651 |
| Random Forest | max_depth = 8, min_samples_leaf = 5 | 0.690 |

Test-set performance

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 72.8% | 68.5% |
| Lapse precision | 0.64 | 0.59 |
| Lapse recall | 0.65 | 0.61 |
| Lapse F1-score | 0.65 | 0.60 |

Confusion matrices

Rows represent actual classes and columns represent predicted classes. Class 1 represents lapsed customers.

Logistic Regression:

| | Predicted non-lapsed | Predicted lapsed |
|---|---:|---:|
| Actual non-lapsed | 111 | 32 |
| Actual lapsed | 31 | 58 |

Random Forest:

| | Predicted non-lapsed | Predicted lapsed |
|---|---:|---:|
| Actual non-lapsed | 105 | 38 |
| Actual lapsed | 35 | 54 |

The majority-class baseline achieved approximately 61.6% accuracy but a lapse-class F1-score of 0.00.

### 7. Training-Set Oversampling

Applied oversampling to the Logistic Regression training data to address class imbalance.

| Metric | Before oversampling | After oversampling |
|---|---:|---:|
| Accuracy | 72.8% | 71.6% |
| Lapse precision | 0.64 | 0.59 |
| Lapse recall | 0.65 | 0.85 |
| Lapse F1-score | 0.65 | 0.697 |

The training set initially contained 570 non-lapsed and 354 lapsed customers. Oversampling balanced these classes to 570 each.

The oversampled model improved lapse recall and F1-score, although precision and overall accuracy decreased slightly.

Selected model: Oversampled Logistic Regression.

This model was selected because identifying more potentially lapsed customers was important for the retention use case. Its higher recall comes with more false-positive predictions, so retention campaigns should not treat every prediction as a confirmed lapse.

Oversampling was applied only to training data.

### 8. Feature Importance and Coefficients

Random Forest top five feature importances:

| Feature | Importance |
|---|---:|
| Frequency | 0.2685 |
| Monetary | 0.2423 |
| Recency | 0.1130 |
| Average delivery minutes | 0.0651 |
| Cancellation rate | 0.0566 |

The three most important features were frequency, monetary value and recency.

Logistic Regression coefficients

Largest positive coefficients:

| Feature | Coefficient |
|---|---:|
| Monetary | +0.1612 |
| Gender: unknown | +0.1265 |
| Recency | +0.1101 |

Largest negative coefficients:

| Feature | Coefficient |
|---|---:|
| Frequency | -1.4652 |
| Favourite category: Electronics | -0.8289 |
| City: Delhi | -0.4953 |

Positive coefficients increase the model's predicted log-odds of the lapse class, while negative coefficients decrease them, holding other model inputs constant. These are model associations, not evidence of causal relationships.

### 9. Retention List

Generated `outputs/retention_list.csv` for the 138 customers in the Premium segment.

- Total Premium customers: 138.
- Premium customers labelled lapsed: 44.
- The list is ordered by predicted lapse probability.

Top 10 customers:

| Customer ID | Predicted lapse probability | Actually lapsed |
|---|---:|---:|
| C0060 | 0.7838 | No |
| C0577 | 0.7742 | No |
| C0984 | 0.7640 | No |
| C0136 | 0.7310 | No |
| C1044 | 0.7265 | No |
| C0512 | 0.7152 | Yes |
| C0082 | 0.7069 | No |
| C0717 | 0.7018 | No |
| C0267 | 0.6917 | No |
| C1032 | 0.6822 | No |

### 10. Model Persistence and Leakage Prevention

Saved the selected Logistic Regression pipeline as `outputs/lapse_model.joblib`.

Reloaded the saved model and verified that its predictions matched the original model predictions.

Leakage prevention measures:

- Split the training and testing datasets before model fitting.
- Fitted preprocessing and scaling within the training pipeline.
- Performed oversampling only on training data.
- Excluded customer IDs and target labels from model features.
- Excluded information from the September 2025 outcome period from historical features.
- Did not use customer segment labels as lapse-model features.

Limitation:The Premium segment was generated using RFM data covering the wider dataset period, whereas the lapse label uses a September 2025 window. Consequently, the Premium retention list is retrospective and should not be interpreted as a leakage-free prospective evaluation.

## Part C 

### 1. Privacy and Input Guardrails

Processed 164 synthetic support tickets using the offline `mock_llm.py` model.

Implemented three PII redaction patterns:

- Email addresses were replaced with `[EMAIL]`.
- Phone numbers were replaced with `[PHONE]`.
- Order IDs were replaced with `[ORDER_ID]`.

Privacy audit results:

| Identifier | Substitutions |
|---|---:|
| Email addresses | 55 |
| Phone numbers | 91 |
| Order IDs | 132 |
| Total | 278 |

A second scan of the redacted text found zero remaining matches for the three targeted identifier patterns.

Additional input guardrails:

| Input check | Tickets |
|---|---:|
| Total tickets | 164 |
| Empty tickets skipped | 4 |
| Tickets exceeding 400 tokens | 4 |
| Tickets eligible for classification | 156 |

Empty and long tickets were not sent for classification. Assertions were also included to verify that the targeted sensitive identifiers do not enter model prompts.

### 2. Classification and Output Validation

Implemented classification using four permitted categories:

- `billing`
- `delivery`
- `technical`
- `fraud`

The system first attempts to parse the model response as JSON. If JSON parsing fails, it extracts category keywords using a regular expression. A keyword classification is accepted only when exactly one permitted category is found.

Invalid outputs are flagged and routed to human review instead of assigning an uncertain category.

### 3. Retry Mechanism

Implemented exponential backoff with jitter.

- Initial model call followed by a maximum of three retries.
- Base delay: 0.05 seconds.
- Random jitter multiplier: 0.9 to 1.15.
- Transient failures are handled using `mock_llm.TransientLLMError`.

During the full comparison runs, all 19 simulated transient errors were recovered successfully.

### 4. Zero-shot and Few-shot Evaluation

Compared zero-shot and few-shot classification using the same synthetic ticket dataset.

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Accuracy | 85.71% | 100.00% |
| Total tokens | 8,308 | 15,796 |
| Tokens per classified ticket | 56.52 | 107.46 |

The zero-shot confusion table showed 11 fraud tickets and 10 technical tickets incorrectly classified as billing.

Few-shot classification improved observed accuracy by 14.29 percentage points but required approximately 90% more tokens per classified ticket.

For this synthetic evaluation, the additional prompt cost improved classification accuracy. However, these results come from a deterministic mock model and should not be treated as a guarantee of real-world model performance.

### 5. Token Budget Demonstration

Implemented a separate zero-shot budget demonstration using a 6,000-token budget.

| Metric | Result |
|---|---:|
| Configured budget | 6,000 tokens |
| Actual tokens used | 5,864 |
| Tickets stopped by budget | 46 |

The system estimates prompt tokens and reserves an additional 100 tokens for the completion before attempting a call. It also checks actual token usage after the call.

The results are saved in `outputs/triage_budget_demo.csv`.

This is an estimated budget-control mechanism. Actual model completion lengths can vary, so the approach is not a universal guarantee against budget overruns.

### 6. Final Few-shot Run and Routing

The final few-shot run uses a token budget of 100,000 and produces `outputs/triage_results.csv`.

Final classification status counts:

| Status | Tickets |
|---|---:|
| classified_json | 135 |
| classified_keyword | 12 |
| flagged_invalid_output | 9 |
| skipped_empty | 4 |
| flagged_long | 4 |
|Total | 164 |

The output CSV contains exactly eight columns:

`ticket_id`, `customer_id`, `status`, `category`, `segment_name`, `priority`, `route`, and `tokens_used`.

Routing results:

| Route | Tickets |
|---|---:|
| support_queue | 102 |
| security_team | 34 |
| human_review | 17 |
| premium_desk | 11 |
|Total | 164 |

Routing rules:

- Empty, long, invalid-output and budget-exhausted tickets are sent to `human_review` with NORMAL priority.
- Valid fraud classifications are sent to `security_team` with HIGH priority.
- Other Premium customer tickets are sent to `premium_desk` with HIGH priority.
- Remaining tickets are sent to `support_queue` with NORMAL priority.

The automated routing validation reported zero errors for the human-review and fraud-routing rules.

### 7. Output Files

- `outputs/tickets_redacted.csv`: ticket data containing redacted text and input status.
- `outputs/triage_budget_demo.csv`: zero-shot token-budget demonstration.
- `outputs/triage_results.csv`: final few-shot classification and routing results.

Limitations: The classifier uses a deterministic mock model and a limited set of synthetic examples. The redaction patterns cover only the three targeted identifier types and do not guarantee detection of every possible form of personal information. Human review remains necessary for invalid and uncertain outputs.


## Part D 

### 1. Policy Document Retrieval

Implemented a Retrieval-Augmented Generation (RAG) assistant using five policy documents:

- `refund_policy.txt`
- `delivery_policy.txt`
- `escalation_policy.txt`
- `payment_policy.txt`
- `privacy_policy.txt`

The documents were divided into chunks of 40 words with a 10-word overlap, using a 30-word step.

| Policy document | Chunks |
|---|---:|
| Refund | 4 |
| Delivery | 3 |
| Escalation | 3 |
| Payment | 3 |
| Privacy | 2 |
| **Total** | **15** |

Used TF-IDF vectorization and cosine similarity to retrieve the three most relevant policy chunks.

For the query `refund policy late delivery`, the top three retrieved chunks came from `refund_policy.txt`.

**TF-IDF explanation:** Term Frequency measures how often a word appears in a chunk. Inverse Document Frequency gives more importance to words that appear in fewer chunks. Their combination helps identify policy passages relevant to a question.

### 2. RAG Evaluation

Evaluated the assistant against 14 questions in `policy_eval.csv`, consisting of 12 answerable questions and 2 unanswerable questions.

| Evaluation metric | 10-word overlap | No overlap |
|---|---:|---:|
| Expected document in top 3 | 12/12 | 12/12 |
| Expected fact in answer | 12/12 | 10/12 |
| Correct fallback answers | 2/2 | 2/2 |

The 10-word overlap configuration achieved all expected evaluation results. Without overlap, two expected facts were missed. This demonstrates how splitting related information across chunk boundaries can affect retrieval and answer generation.

Limitation: TF-IDF is based on word matching rather than semantic understanding. Questions using different terminology may retrieve less relevant passages.

### 3. Refund Agent and Tool Architecture

Implemented a ReAct-style refund agent using the offline `mock_llm.py` model.

The agent uses five tools:

| Tool | Purpose |
|---|---|
| `get_order` | Retrieves order details from SQLite |
| `search_policy` | Retrieves relevant policy chunks |
| `calculate_refund` | Calculates the eligible refund amount |
| `issue_refund` | Records an automatically approved refund |
| `escalate_to_human` | Records cases requiring human approval |

The agent checks order existence, completion status, delivery time and refund amount. Automatic refunds are limited to INR 1,000. Refunds above INR 1,000 are sent for human approval.

The agent retrieves order information from `data/instamart.db` and records refund and escalation actions in `outputs/refund_ledger.jsonl`.

### 4. Scenario Evaluation

Executed seven supplied scenarios.

| Scenario | Expected result | Actual result | Refund amount (INR) | Model calls | Tokens |
|---|---|---|---:|---:|---:|
| S1 | REFUND | REFUND | 128.25 | 5 | 1,034 |
| S2 | REFUND | REFUND | 135.00 | 5 | 1,034 |
| S3 | ESCALATE | ESCALATE | 1,999.00 | 5 | 1,024 |
| S4 | DENIED | DENIED | 0 | 4 | 706 |
| S5 | DENIED | DENIED | 0 | 3 | 406 |
| S6 | NOT_FOUND | NOT_FOUND | 0 | 2 | 109 |
| S7 | FALLBACK_MAX_ITERATIONS | FALLBACK_MAX_ITERATIONS | 0 | 6 | 3,162 |

All seven scenarios produced their expected statuses.

The results are saved in `outputs/agent_runs.jsonl`. Refund and escalation actions are recorded in `outputs/refund_ledger.jsonl`.

### 5. Token and Time Limits

Implemented safeguards to prevent the agent from continuing indefinitely.

| Test | Configuration | Result |
|---|---|---|
| Token limit | 200 tokens | FALLBACK_TOKEN_LIMIT |
| Time limit | 0 seconds | FALLBACK_TIMEOUT |
| Maximum iterations | 6 | FALLBACK_MAX_ITERATIONS in S7 |

The default agent configuration uses a maximum of six iterations, a 6,000-token limit and a 10-second time limit.

When a limit is reached, the agent returns a fallback response directing the request to human support.

### 6. Checkpoint and Recovery

The agent saves checkpoint information after executing tools and recording their observations.

Checkpoint files are stored in the `state/` directory and contain:

- Conversation messages.
- Token usage.
- Model call count.
- Executed tool names.
- Tool arguments and observations.
- Current step number.

A simulated crash was triggered during S1 after two tool executions.

The agent was then resumed using the saved checkpoint.

| Recovery metric | Result |
|---|---|
| Simulated crash | After two tools |
| Final status | REFUND |
| Final refund amount | INR 128.25 |
| Total model calls | 5 |
| Tools executed | get_order, search_policy, calculate_refund, issue_refund |

The resumed agent completed the refund without repeating previously executed tools.

### 7. ReAct versus Plan-and-Execute

ReAct was selected because it chooses the next action based on the latest tool observation.

This is useful when the agent encounters unexpected results, such as a missing order, cancelled order or refund requiring human approval.

In the tested scenarios, S1 required five model calls and 1,034 tokens, S5 required three calls and 406 tokens, and S6 required two calls and 109 tokens.

A fixed plan can be simpler for predictable workflows, but it may not adapt as naturally to unexpected tool results. ReAct was selected for its ability to adjust the next action based on observations.

Limitations: The agent uses a deterministic mock model and synthetic data. The seven-day eligibility helper is defined separately but is not invoked by the scripted scenario flow. The supplied scenarios are treated as requests within the policy window. A production implementation would also require stronger transaction handling, duplicate-action protection, authorization and explicit policy-window enforcement.

## Limitations 

- Synthetic data: The project uses generated data and does not represent real Instamart customers, transactions, or operational policies.
- Mock language model: LLM behaviour is simulated using the supplied `mock_llm.py`; results may differ from a production language model.
- Lapse prediction: The model is trained and evaluated on a synthetic historical dataset. Its predictions should not be treated as validated real-world customer behaviour.
- Retention list: The Premium customer retention list is illustrative and retrospective, not a leakage-free prospective campaign.
- Privacy guardrails:PII redaction targets the specified email, phone, and order ID patterns; it is not a complete general-purpose privacy solution.
- Policy retrieval:TF-IDF retrieval uses keyword similarity and may miss relevant information when wording differs from the policy documents.
- Refund agent: The agent uses predefined tools and synthetic scenarios. It is not connected to a real payment or refund system.

