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

Create a task file (for example `inspect_tasks/fraud_triage.py`) that reuses the
helper and emits structured findings for scoring:

```python
import pandas as pd
from inspect_ai import Task, task
from inspect_ai.dataset import dataset
from inspect_ai.result import JSONResult

from app import review_claims_with_chatlas


@dataset
def january_claims():
   df = pd.read_csv("sample_data/january_claims.csv")
   for record in df.to_dict("records"):
      yield record


@task
def fraud_triage() -> Task:
   claims = list(january_claims())
   llm_result = review_claims_with_chatlas(claims)

   return JSONResult(
      data=llm_result.flagged_claims,
      extras={"summary": llm_result.summary, "raw_text": llm_result.raw_text},
   )
```

You can then launch Inspect with `inspect run inspect_tasks/fraud_triage.py`
and add additional datasets (for example `sample_data/february_claims.csv` or
`sample_data/flagged_examples.csv`) to expand coverage. The task reuses the same
prompt and parsing logic as the Shiny app, so UI results and automated checks
stay aligned.

Open the provided URL and upload a claims CSV. Common columns such as `claim_id`, `policy_id`, `claim_date`, and `claim_amount` help the LLM reason effectively, but you can include any additional fields that may aid the investigation. The summary panel reports the Bedrock response text or any connectivity errors.
