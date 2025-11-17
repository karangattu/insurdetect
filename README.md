# InsurDetect — Insurance Claim Fraud Triage

InsurDetect is a small repo that demonstrates a guardrailed, LLM-backed fraud triage workflow for insurance claims and shows how to use Inspect AI with Bedrock models to evaluate claim data.

## Quick Start

1. Install uv if not installed

```bash
pip install uv
```

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

1. Set environment variables for Bedrock/AWS credentials:

```bash
export AWS_REGION=us-east-1
```

1. Run the fraud triage evaluation using the Inspect AI task (example):

```bash
inspect eval detect_fraud_eval.py
```
