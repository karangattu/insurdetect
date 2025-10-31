# InsurDetect Fraud Review

Shiny for Python dashboard that triages monthly insurance claims and routes
them to AWS Bedrock (Anthropic Claude) for fraud risk scoring.

## Features

- Upload monthly claim batches as CSV files.
- Raw rows are forwarded to Anthropic Claude via `chatlas.ChatBedrockAnthropic`; the
  LLM performs all fraud heuristics.
- Shiny UI surfaces the LLM verdicts, severity, and rationale with Font Awesome
  icons from [`faicons`](https://pypi.org/project/faicons/).

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   uv venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Optionally override the default model and endpoint:
   - `BEDROCK_MODEL_ID` (defaults to `us.anthropic.claude-haiku-4-5-20251001-v1:0`).

## Running the app

Launch the dashboard with Shiny for Python:

```bash
shiny run --reload app:app
```

## Programmatic use

For evaluations or automated testing (for example with `inspect-ai`), call
`app.review_claims_with_chatlas(...)` directly with a pandas DataFrame or a list
of dictionaries to obtain the parsed `LLMResult` without running the Shiny UI.

### Quick start helper

```python
import pandas as pd

from app import build_system_prompt, review_claims_with_chatlas

df = pd.read_csv("sample_data/january_claims.csv")

result = review_claims_with_chatlas(
   claims=df.to_dict("records"),
   system_prompt=build_system_prompt(),
)

print(result.summary)
for flagged in result.flagged_claims:
   print(flagged["claim_id"], flagged["llm_flag_reason"])
```

### inspect-ai evaluation task

Create a task file (for example `inspect_tasks/fraud_triage.py`) that uses
`chatlas.to_solver()` to integrate with inspect-ai:

```python
import pandas as pd
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, FieldSpec, json_dataset
from inspect_ai.scorer import Score, scorer, accuracy, Target

from app import build_system_prompt, make_chat_client


def load_claims_dataset(path: str):
    """Load claims CSV and convert to inspect-ai Sample format."""
    df = pd.read_csv(path)
    samples = []
    
    for _, record in df.iterrows():
        # Assume CSV has 'is_fraud' column for ground truth
        samples.append(
            Sample(
                input=str(record.to_dict()),
                target=str(record.get("is_fraud", False)),
                metadata={
                    "claim_id": record.get("claim_id"),
                    "claim_amount": record.get("claim_amount"),
                    "policy_id": record.get("policy_id"),
                },
            )
        )
    
    return samples


@scorer(metrics=[accuracy()])
def fraud_detection_scorer():
    """Score fraud predictions against ground truth labels."""

    async def score(state, target: Target):
        # The solver's output is in state.output.completion
        output_text = state.output.completion.lower()
        
        # Simple check: did the model flag this as fraud?
        predicted_fraud = "fraud" in output_text or "flagged" in output_text
        actual_fraud = target.text.lower() == "true"
        
        is_correct = predicted_fraud == actual_fraud
        
        return Score(
            value=is_correct,
            answer=str(predicted_fraud),
            explanation=f"Predicted fraud: {predicted_fraud}, Actual: {actual_fraud}",
        )

    return score


@task
def fraud_triage():
    """Evaluate fraud detection against labeled claims data."""
    # Create chatlas client with system prompt and tools
    chat = make_chat_client(system_prompt=build_system_prompt())
    
    # Convert chatlas client to inspect-ai solver
    # This preserves the system prompt, tools, and model config
    solver = chat.to_solver()
    
    return Task(
        dataset=load_claims_dataset("sample_data/january_claims.csv"),
        solver=solver,
        scorer=fraud_detection_scorer(),
    )
```

**Running the evaluation:**

```bash
# Run with default settings
inspect eval inspect_tasks/fraud_triage.py

# View results interactively
inspect view

# Run with specific model override
inspect eval inspect_tasks/fraud_triage.py \
  --model bedrock/anthropic.claude-3-sonnet-20240229-v1:0

# Run with custom dataset
inspect eval inspect_tasks/fraud_triage.py \
  -T dataset=sample_data/flagged_examples.csv
```

**Key components:**

- **chatlas integration**: Use `chat.to_solver()` to convert your chatlas client (with all its tools, prompts, and config) into an inspect-ai solver
- **Dataset with targets**: Each `Sample` includes the claim data as `input` and ground truth fraud label as `target`
- **Custom scorer**: Compares LLM predictions against ground truth and computes accuracy
- **Tool preservation**: The `get_policy_statistics` tool registered in your chat client is automatically available during evaluation

**For more advanced evals**, you can use `chat.export_eval()` to capture multi-turn conversations as JSONL datasets:

```python
# Build a conversation with the LLM
chat.chat("Review this claim: claim_id=C-1001, amount=$50000")
chat.chat("Is this suspicious?")

# Export as eval dataset with custom grading criteria
chat.export_eval(
    "datasets/claim_review.jsonl",
    target="Response should flag high-dollar claims and cite policy statistics",
)
```

The task reuses the same prompt, tools, and model config as the Shiny app, ensuring UI results and automated evaluations stay perfectly aligned.

Open the provided URL and upload a claims CSV. Common columns such as `claim_id`, `policy_id`, `claim_date`, and `claim_amount` help the LLM reason effectively, but you can include any additional fields that may aid the investigation. The summary panel reports the Bedrock response text or any connectivity errors.
