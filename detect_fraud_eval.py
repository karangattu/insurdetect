from inspect_ai import Task, task
from inspect_ai.dataset import csv_dataset
from inspect_ai.scorer import model_graded_qa
from shiny_app.chat_utils import create_chat

chat = create_chat()


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
