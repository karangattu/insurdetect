# InsurDetect — Insurance Claim Fraud Triage

InsurDetect is a small repo that demonstrates a guardrailed, LLM-backed fraud triage workflow for insurance claims and shows how to use Inspect AI with Bedrock models to evaluate claim data.

## Quick Start

1. Install uv if not installed

```bash
pip install uv
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Set environment variables for Bedrock/AWS credentials:

```bash
export AWS_REGION=us-east-1
```

4. Run the fraud triage evaluation using the Inspect AI task (example):

```bash
inspect eval detect_fraud_eval.py
```

5. View the results in a web browser:

```bash
inspect view
```

This will start a local server running on `http://127.0.0.1:7575`.

6. Create a HTML report of the evaluation:

```bash
inspect view bundle --output-dir reports
```

open the index.html file in the reports directory in a web browser to see the report.
