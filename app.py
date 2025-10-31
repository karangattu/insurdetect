from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import pandas as pd
from chatlas import ChatBedrockAnthropic
from faicons import icon_svg
from shiny import App, Inputs, Outputs, Session, reactive, render, req, ui

APP_TITLE = "InsurDetect Fraud Review"


def build_system_prompt() -> str:
    """Return the default guardrailed instructions for the Bedrock chat."""

    return (
        "You are a senior insurance fraud investigator. "
        "Triage claim rows and assign a fraud risk rating. "
        "Follow these guardrails:\n"
        "1. Flag claims filed less than 15 calendar days "
        "after the prior claim.\n"
        "2. Flag claims that exceed $25,000 or are over twice the policy's "
        "recent average payout.\n"
        "3. Flag duplicate submissions for the same policy date.\n"
        "4. Flag narratives containing high risk terms such as 'staged', "
        "'exaggerated', 'repeat', or 'police report pending'.\n"
        "Explain each flag, rate severity as low, medium, or high, and "
        "return JSON that matches the caller's schema. Rely solely on the "
        "provided data."
    )


SYSTEM_PROMPT = build_system_prompt()

ICON_POSITIVE = icon_svg("circle-check", fill="#198754")
ICON_ALERT = icon_svg("triangle-exclamation", fill="#dc3545")
ICON_INFO = icon_svg("circle-info", fill="#0d6efd")


@dataclass
class LLMResult:
    flagged_claims: List[Dict[str, Any]]
    summary: str
    raw_text: str
    error: Optional[str] = None


def make_chat_client(system_prompt: str = SYSTEM_PROMPT) -> Any:
    """Create a Bedrock-backed chat client using environment defaults."""

    model_id = os.getenv(
        "BEDROCK_MODEL_ID",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    )
    client_kwargs: Dict[str, Any] = {
        "model": model_id,
        "system_prompt": system_prompt,
    }

    env_to_param = {
        "AWS_REGION": "aws_region",
        "AWS_PROFILE": "aws_profile",
        "AWS_ACCESS_KEY_ID": "aws_access_key",
        "AWS_SECRET_ACCESS_KEY": "aws_secret_key",
        "AWS_SESSION_TOKEN": "aws_session_token",
    }

    for env_key, param_key in env_to_param.items():
        value = os.getenv(env_key)
        if value:
            client_kwargs[param_key] = value

    base_url = os.getenv("BEDROCK_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url

    return ChatBedrockAnthropic(**client_kwargs)


def extract_text(response: Any) -> str:
    """Normalize chat output to plain text."""

    if isinstance(response, str):
        return response

    if isinstance(response, dict):
        text_value = response.get("text") or response.get("message") or ""
        if text_value:
            return text_value

    for attr in ("text", "message", "content", "content_text"):
        if hasattr(response, attr):
            candidate = getattr(response, attr)
            if isinstance(candidate, str):
                return candidate
            if isinstance(candidate, Iterable):
                parts = [part for part in candidate if isinstance(part, str)]
                if parts:
                    return "\n".join(parts)

    return str(response)


def parse_llm_output(payload: str) -> LLMResult:
    """Extract structured content from the language model response."""

    cleaned = payload.strip()
    json_start = cleaned.find("{")
    json_end = cleaned.rfind("}")
    if json_start == -1 or json_end == -1:
        return LLMResult(
            flagged_claims=[],
            summary="",
            raw_text=payload,
            error="No JSON object found",
        )

    json_blob = cleaned[json_start:json_end + 1]
    try:
        parsed = json.loads(json_blob)
    except json.JSONDecodeError as exc:
        return LLMResult(
            flagged_claims=[],
            summary="",
            raw_text=payload,
            error=f"Failed to decode JSON: {exc}",
        )

    flagged = parsed.get("flagged_claims", [])
    summary = parsed.get("summary", "")
    return LLMResult(flagged_claims=flagged, summary=summary, raw_text=payload)


def review_claims_with_chatlas(
    claims: Union[pd.DataFrame, Sequence[Mapping[str, Any]]],
    *,
    client: Optional[Any] = None,
    system_prompt: str = SYSTEM_PROMPT,
    request_id: Optional[str] = None,
) -> LLMResult:
    """Submit claim rows to ChatBedrockAnthropic and parse the response.

    This helper isolates the Chatlas call so that it can be exercised directly
    in inspections and automated evaluations. When *client* is ``None`` the
    default Bedrock client is constructed via :func:`make_chat_client`.
    """

    if isinstance(claims, pd.DataFrame):
        records = claims.to_dict(orient="records")
        columns = list(claims.columns)
    else:
        records = []
        for row in claims:
            if isinstance(row, Mapping):
                row_dict = dict(row)
            elif hasattr(row, "to_dict"):
                row_dict = row.to_dict()
            else:
                row_dict = dict(row)
            records.append(row_dict)
        columns = list(records[0].keys()) if records else []

    if not records:
        return LLMResult(
            flagged_claims=[],
            summary="No rows supplied",
            raw_text="",
        )

    prompt_payload = {
        "request_id": request_id or str(uuid.uuid4()),
        "instructions": (
            "Review the provided insurance claims and decide which rows are "
            "potential fraud. Return JSON with 'flagged_claims' (list) and "
            "'summary' (string)."
        ),
        "columns": columns,
        "claims": records,
        "response_schema": {
            "flagged_claims": [
                {
                    "row_index": "int",
                    "claim_id": "string",
                    "severity": "one of ['low','medium','high']",
                    "reasons": "list of brief explanations",
                }
            ],
            "summary": "string",
        },
    }

    user_prompt = (
        "Use only the supplied data. Row indices are zero-based. "
        "Return JSON that matches the schema exactly.\n"
        f"{json.dumps(prompt_payload, indent=2)}"
    )

    chat_client = client or make_chat_client(system_prompt=system_prompt)

    try:
        response = chat_client.chat(user_prompt)
        text = extract_text(response)
        return parse_llm_output(text)
    except Exception as exc:  # noqa: BLE001
        return LLMResult(
            flagged_claims=[],
            summary="",
            raw_text="",
            error=f"Bedrock request failed: {exc}",
        )


def call_bedrock(df: pd.DataFrame) -> LLMResult:
    return review_claims_with_chatlas(df)


def load_claims(file_path: str, file_name: str) -> pd.DataFrame:
    _, ext = os.path.splitext(file_name.lower())
    if ext != ".csv":
        raise ValueError("Unsupported file type. Upload a CSV file.")
    return pd.read_csv(file_path)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    cleaned.reset_index(drop=True, inplace=True)
    return cleaned


def merge_llm_flags(df: pd.DataFrame, llm_result: LLMResult) -> pd.DataFrame:
    merged = df.copy()
    lookup = {
        int(item.get("row_index", -1)): item for item in llm_result.flagged_claims
    }
    merged["llm_flagged"] = merged.index.map(lambda idx: idx in lookup)
    merged["llm_reasons"] = merged.index.map(
        lambda idx: lookup[idx].get("reasons", []) if idx in lookup else []
    )
    merged["severity"] = merged.index.map(
        lambda idx: lookup[idx].get("severity", "") if idx in lookup else ""
    )
    return merged


def format_table(df: pd.DataFrame) -> str:
    display_df = df.copy()
    base_cols = [
        col
        for col in display_df.columns
        if col not in {"llm_flagged", "llm_reasons", "severity"}
    ]

    if "claim_date" in display_df.columns:
        display_df["claim_date"] = display_df["claim_date"].astype(str)

    if "claim_amount" in display_df.columns:
        numeric_amounts = pd.to_numeric(
            display_df["claim_amount"],
            errors="coerce",
        )
        display_df["claim_amount"] = numeric_amounts.map(
            lambda value: f"${value:,.2f}" if pd.notna(value) else ""
        )

    display_df["Flagged"] = display_df["llm_flagged"].map(
        lambda flagged: ICON_ALERT if flagged else ICON_POSITIVE
    )
    display_df["Severity"] = display_df["severity"].apply(
        lambda value: value.title() if isinstance(value, str) and value else ""
    )
    display_df["LLM reasons"] = display_df["llm_reasons"].apply(
        lambda reasons: "<br/>".join(reasons) if reasons else ""
    )

    table_columns = ["Flagged", "Severity", *base_cols, "LLM reasons"]
    table_df = display_df[table_columns]
    return table_df.to_html(
        classes="table table-hover",
        index=False,
        escape=False,
    )


def server(input: Inputs, output: Outputs, session: Session) -> None:
    @reactive.Calc
    def claims_df() -> pd.DataFrame:
        uploads = input.claims_file()
        req(uploads)
        file_info = uploads[0]
        df = load_claims(file_info["datapath"], file_info["name"])
        return clean_columns(df)

    @reactive.Calc
    def llm_result() -> LLMResult:
        return call_bedrock(claims_df())

    @reactive.Calc
    def consolidated() -> pd.DataFrame:
        return merge_llm_flags(claims_df(), llm_result())

    @output
    @render.ui
    def guidance() -> ui.Tag:
        return ui.div(
            {"class": "help-panel"},
            ui.HTML(
                f"{ICON_INFO} Upload a CSV file of claims. "
                "Fraud screening is handled entirely by the Bedrock LLM."
            ),
        )

    @output
    @render.ui
    def summary_cards() -> ui.Tag:
        uploads = input.claims_file()
        if not uploads:
            return ui.div()

        df = consolidated()
        llm = llm_result()
        total = len(df)
        flagged_total = int(df["llm_flagged"].sum())
        cleared_total = total - flagged_total

        cards = ui.div(
            {"class": "metrics"},
            ui.div(
                {"class": "metric"},
                ui.HTML(f"{ICON_INFO} Total claims"),
                ui.h2(str(total)),
            ),
            ui.div(
                {"class": "metric"},
                ui.HTML(f"{ICON_ALERT} Flagged by LLM"),
                ui.h2(str(flagged_total)),
            ),
            ui.div(
                {"class": "metric"},
                ui.HTML(f"{ICON_POSITIVE} Cleared"),
                ui.h2(str(cleared_total)),
            ),
        )

        detail: List[ui.Tag] = []
        if flagged_total:
            severity_counts = df[df["llm_flagged"]]["severity"].value_counts().to_dict()
            severity_text = ", ".join(
                f"{level.title()}: {count}"
                for level, count in severity_counts.items()
                if level
            )
            if severity_text:
                detail.append(ui.p(ui.strong("Flag severities:"), " ", severity_text))
        if llm.summary:
            detail.append(ui.p(ui.strong("LLM summary:"), " ", llm.summary))
        if llm.error:
            detail.append(ui.p(ui.strong("LLM status:"), " ", llm.error))

        return ui.div(cards, *detail)

    @output
    @render.ui
    def results_table() -> ui.Tag:
        uploads = input.claims_file()
        if not uploads:
            return ui.div("Upload a CSV file to begin.")

        df = consolidated()
        return ui.div(ui.HTML(format_table(df)))


app_ui = ui.page_fluid(
    ui.tags.style(
        """
        body {
            background-color: #f5f6fa;
            font-family: 'Helvetica Neue', Arial, sans-serif;
        }
        .app-title {margin-bottom: 0.25rem;}
        .help-panel {margin-top: 0.5rem; color: #495057;}
        .metrics {display: flex; gap: 1rem; flex-wrap: wrap; margin: 1.5rem 0;}
        .metric {
            background: white;
            border-radius: 0.75rem;
            padding: 1rem;
        }
        .metric h2 {margin: 0.25rem 0 0 0;}
        table {background: white; border-radius: 0.5rem; overflow: hidden;}
        th {background: #343a40; color: white;}
        tbody tr:nth-child(even) {background: #f1f3f5;}
        tbody tr:hover {background: #e9ecef;}
        """
    ),
    ui.panel_title(APP_TITLE),
    ui.row(
        ui.column(
            6,
            ui.input_file(
                "claims_file",
                "Upload monthly claim data (CSV)",
                multiple=False,
                accept=[".csv"],
            ),
        ),
    ),
    ui.output_ui("guidance"),
    ui.output_ui("summary_cards"),
    ui.h3("Claim review results"),
    ui.output_ui("results_table"),
)

app = App(app_ui, server)

if __name__ == "__main__":
    app.run()
