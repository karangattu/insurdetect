from chatlas import ChatBedrockAnthropic
from inspect_ai import Task, task
from inspect_ai.dataset import csv_dataset
from inspect_ai.scorer import model_graded_qa


def check_if_future_date(claim_date: str, today_date: str) -> bool:
    """Check if the claim date is in the future compared to today's date."""
    from datetime import datetime

    claim_dt = datetime.fromisoformat(claim_date).date()
    today_dt = datetime.fromisoformat(today_date).date()
    return claim_dt > today_dt


def build_system_prompt() -> str:
    """Return the default guardrailed instructions for the Bedrock chat."""

    prompt = """
You are a senior insurance fraud investigator. Triage claim rows and assign a fraud risk rating. Follow these guardrails:

1. Flag claims filed ≤15 calendar days after the prior claim.
2. Flag claims > $25,000.
3. Flag duplicate submissions for the same policy date.
4. Flag narratives containing high‑risk terms: "staged", "exaggerated", "repeat", "police report pending".
5. Flag claims that are in the future (after today's date).
6. If a claim is flagged, provide detailed reasons for each flag.
7. If a claim is not flagged, respond with "none" and payment as "auto" otherwise payment is "pending".

Only raise flags when a listed guardrail is triggered. If none apply,
return an empty flags array, set overall_severity to "none", and
payment to "auto". When at least one guardrail triggers, provide a flag
entry per issue and set payment to "pending".

Explain each flag, rate severity as low, medium, or high, and return JSON
as shown below. Rely solely on the provided data.

        Input JSON schema (example):

        {
            "claim_id": "C-2001",
            "policy_id": "P-001",
            "claim_date": "2025-02-01",
            "claim_amount": 28000,
            "claim_type": "auto",
            "description": "Rear-end collision; police report filed; staged
            accident suspected."
        }


        Output JSON schema (example):
        {
            "claim_id": "string",
            "flags": [
                {
                    "reason": "string",
                    "severity": "low|medium|high"
                }
            ],
            "overall_severity": "low|medium|high|none",
            "payment": "auto|pending"
        }
"""
    return prompt


chat = ChatBedrockAnthropic(
    model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    cache="none",
    aws_region="us-east-1",
    system_prompt=build_system_prompt(),
)

chat.register_tool(check_if_future_date)


@task
def insurance_fraud_detection():
    """Evaluate insurance claims for fraud detection."""
    return Task(
        dataset=csv_dataset("claims_list.csv"),
        solver=chat.to_solver(),
        scorer=model_graded_qa(),
        name="insurance_fraud_detection_february_claims",
        model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    )
