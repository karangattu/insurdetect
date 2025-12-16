from chatlas import ChatBedrockAnthropic
from inspect_ai import Task, task
from inspect_ai.dataset import csv_dataset
from inspect_ai.scorer import model_graded_qa


def check_if_future_date(claim_date: str, today_date: str | None = None) -> bool:
    """
    Check if claim_date is in the future. If today_date is not provided,
    compute it server-side to avoid model-supplied wrong values.
    """
    from datetime import date, datetime

    try:
        claim_dt = datetime.fromisoformat(claim_date).date()
    except Exception:
        # fallback: try parsing a trimmed or different format
        claim_dt = datetime.strptime(claim_date[:10], "%Y-%m-%d").date()

    if today_date:
        try:
            today_dt = datetime.fromisoformat(today_date).date()
        except Exception:
            today_dt = date.today()
    else:
        today_dt = date.today()

    return claim_dt > today_dt


def build_system_prompt() -> str:
    """Return the default guardrailed instructions for fraud detection."""

    prompt = """You are a senior insurance fraud investigator. Review each insurance claim and assess the fraud risk. Be concise and natural.

IMPORTANT: This system is designed ONLY for insurance claim fraud analysis. If the input:
- Is not an insurance claim (e.g., general questions, unrelated topics)
- Is incomplete or missing key details (claim amount, dates, description)
- Is a test message or greeting

Respond politely: "I'm sorry, but I can only analyze insurance claims. Please provide a claim with: Claim ID, claim amount, policy start date, claim date, and claim description. For example: 'Claim C-2001 for $28,000 filed on Feb 1, 2025 (policy started Dec 15, 2024). Auto claim: Rear-end collision.'"

FRAUD RISK RULES:
1. Claims over $25,000 are medium risk
2. Claims $50,000+ are high risk
3. Claims filed within 30 days of policy start are medium risk
4. Suspicious keywords ("staged", "duplicate", "exaggerated", "repeat", "police report pending") = high risk
5. Future-dated claims = high risk
6. Vague language ("repair estimate high", "few witnesses", "no witnesses") = high risk
7. If multiple medium flags exist (2+), escalate to high risk

INPUT: You'll receive a claim with these details:
- Claim ID, policy start date, claim date, amount, type, and description

OUTPUT: Provide your assessment in natural language:
- Briefly describe any red flags you found (or say "No red flags")
- State the overall risk level: LOW, MEDIUM, HIGH, or NONE
- State the payment decision: AUTO-APPROVE or PENDING
- Keep it to 2-3 sentences max

EXAMPLE INPUT:
Claim C-2001 for $28,000 filed on Feb 1, 2025 (policy started Dec 15, 2024).
Auto claim: Rear-end collision; police report filed; staged accident suspected.

EXAMPLE OUTPUT:
Red flags: High claim amount ($28K) and suspicious term "staged" in description.
Risk: HIGH.
Decision: PENDING review."""
    return prompt


chat = ChatBedrockAnthropic(
    model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
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
        scorer=model_graded_qa(partial_credit=True),
        name="insurance_fraud_detection_february_claims",
        model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    )
