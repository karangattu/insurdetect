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

    if "```json" in cleaned:
        start = cleaned.find("```json") + 7
        end = cleaned.find("```", start)
        if end != -1:
            cleaned = cleaned[start:end].strip()

    json_start = cleaned.find("{")
    json_end = cleaned.rfind("}")
    if json_start == -1 or json_end == -1:
        return LLMResult(
            flagged_claims=[],
            summary="",
            raw_text=payload[:500],  # Truncate for display
            error="No JSON object found",
        )

    json_blob = cleaned[json_start : json_end + 1]

    # Try standard parse first
    try:
        parsed = json.loads(json_blob)
    except json.JSONDecodeError as exc:
        # Try to salvage partial JSON by finding valid flagged_claims array
        try:
            flagged_start = json_blob.find('"flagged_claims"')
            if flagged_start != -1:
                array_start = json_blob.find("[", flagged_start)
                if array_start != -1:
                    # Find matching closing bracket
                    bracket_count = 0
                    array_end = -1
                    for i in range(array_start, len(json_blob)):
                        if json_blob[i] == "[":
                            bracket_count += 1
                        elif json_blob[i] == "]":
                            bracket_count -= 1
                            if bracket_count == 0:
                                array_end = i + 1
                                break

                    if array_end != -1:
                        flagged_array = json.loads(json_blob[array_start:array_end])
                        return LLMResult(
                            flagged_claims=flagged_array,
                            summary="Partial response recovered",
                            raw_text=payload[:500],
                            error=f"Original parse failed, recovered {len(flagged_array)} flags",
                        )
        except Exception:
            pass

        return LLMResult(
            flagged_claims=[],
            summary="",
            raw_text=payload[:500],
            error=f"Failed to decode JSON: {exc}",
        )

    flagged = parsed.get("flagged_claims", [])
    summary = parsed.get("summary", "")
    return LLMResult(flagged_claims=flagged, summary=summary, raw_text=payload[:500])


def get_policy_statistics(
    policy_id: str, claims_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate statistics for a given policy from the claims dataset.

    Parameters
    ----------
    policy_id : The policy identifier to analyze.
    claims_data : The full list of claim records.

    Returns
    -------
    A dict with 'policy_id', 'claim_count', 'avg_amount', 'max_amount'.
    """
    policy_claims = [
        c
        for c in claims_data
        if str(c.get("policy_id", "")).strip() == str(policy_id).strip()
    ]
    if not policy_claims:
        return {
            "policy_id": policy_id,
            "claim_count": 0,
            "avg_amount": 0.0,
            "max_amount": 0.0,
        }

    amounts = [float(c.get("claim_amount", 0)) for c in policy_claims]
    return {
        "policy_id": policy_id,
        "claim_count": len(amounts),
        "avg_amount": sum(amounts) / len(amounts) if amounts else 0.0,
        "max_amount": max(amounts) if amounts else 0.0,
    }


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

    # Process in batches to avoid overwhelming the LLM with large payloads
    BATCH_SIZE = 100
    all_flagged = []
    all_summaries = []

    chat_client = client or make_chat_client(system_prompt=system_prompt)

    # Register the tool so the LLM can request policy statistics
    def tool_get_policy_stats(policy_id: str) -> Dict[str, Any]:
        """
        Get claim statistics for a specific policy.

        Parameters
        ----------
        policy_id : The policy identifier to analyze.
        """
        return get_policy_statistics(policy_id, records)

    chat_client.register_tool(tool_get_policy_stats)

    for batch_start in range(0, len(records), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(records))
        batch_records = records[batch_start:batch_end]

        prompt_payload = {
            "request_id": request_id or str(uuid.uuid4()),
            "batch_info": f"Processing rows {batch_start} to {batch_end - 1} of {len(records)} total",
            "instructions": (
                "Review the provided insurance claims and decide which rows are "
                "potential fraud. Return JSON with 'flagged_claims' (list) and "
                "'summary' (string). Use row indices relative to the full dataset."
            ),
            "columns": columns,
            "claims": batch_records,
            "offset": batch_start,  # Tell LLM the starting row index
            "response_schema": {
                "flagged_claims": [
                    {
                        "row_index": "int (use offset + local index)",
                        "claim_id": "string",
                        "severity": "one of ['low','medium','high']",
                        "reasons": "list of brief explanations",
                    }
                ],
                "summary": "string",
            },
        }

        user_prompt = (
            "Use only the supplied data. Row indices should be offset + local position. "
            "Return JSON that matches the schema exactly.\n"
            f"{json.dumps(prompt_payload, indent=2)}"
        )

        try:
            response = chat_client.chat(user_prompt)
            text = extract_text(response)
            batch_result = parse_llm_output(text)

            if batch_result.flagged_claims:
                all_flagged.extend(batch_result.flagged_claims)
            if batch_result.summary:
                all_summaries.append(batch_result.summary)

        except Exception as exc:  # noqa: BLE001
            all_summaries.append(f"Batch {batch_start}-{batch_end - 1} failed: {exc}")

    combined_summary = (
        "\n".join(all_summaries) if all_summaries else "No issues detected"
    )

    return LLMResult(
        flagged_claims=all_flagged,
        summary=combined_summary,
        raw_text=f"Processed {len(records)} claims in {(len(records) + BATCH_SIZE - 1) // BATCH_SIZE} batches",
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
    def value_boxes() -> ui.TagList:
        uploads = input.claims_file()
        if not uploads:
            return ui.TagList()

        df = consolidated()
        total = len(df)
        flagged_total = int(df["llm_flagged"].sum())
        cleared_total = total - flagged_total
        flagged_pct = (flagged_total / total * 100) if total > 0 else 0

        return ui.TagList(
            ui.value_box(
                title="Total Claims",
                value=str(total),
                showcase=ICON_INFO,
                theme="primary",
            ),
            ui.value_box(
                "Flagged for Review",
                str(flagged_total),
                ui.p(f"{flagged_pct:.1f}% of total claims"),
                showcase=ICON_ALERT,
                theme="danger" if flagged_total > 0 else "secondary",
            ),
            ui.value_box(
                title="Cleared",
                value=str(cleared_total),
                showcase=ICON_POSITIVE,
                theme="success",
            ),
        )

    @output
    @render.ui
    def summary_card() -> ui.Tag:
        uploads = input.claims_file()
        if not uploads:
            return ui.div()

        df = consolidated()
        llm = llm_result()
        flagged_total = int(df["llm_flagged"].sum())

        content: List[ui.Tag] = []

        if flagged_total:
            # Show flagged claim IDs first
            flagged_df = df[df["llm_flagged"]]
            if "claim_id" in flagged_df.columns:
                content.append(ui.h5("Flagged Claims"))
                claim_ids = flagged_df["claim_id"].tolist()

                # Group by severity if available
                if "severity" in flagged_df.columns:
                    severity_groups = {}
                    for _, row in flagged_df.iterrows():
                        sev = row.get("severity", "unknown")
                        cid = row.get("claim_id", "N/A")
                        if sev not in severity_groups:
                            severity_groups[sev] = []
                        severity_groups[sev].append(cid)

                    # Display by severity
                    for severity in ["high", "medium", "low"]:
                        if severity in severity_groups:
                            ids = severity_groups[severity]
                            content.append(
                                ui.tags.div(
                                    {
                                        "class": f"alert alert-{'danger' if severity == 'high' else 'warning' if severity == 'medium' else 'info'} mb-2"
                                    },
                                    ui.strong(
                                        f"{severity.title()} Risk ({len(ids)}): "
                                    ),
                                    ", ".join(str(cid) for cid in ids[:20]),
                                    f" ... and {len(ids) - 20} more"
                                    if len(ids) > 20
                                    else "",
                                )
                            )
                else:
                    # No severity info, just show all claim IDs
                    content.append(
                        ui.tags.div(
                            {"class": "alert alert-warning mb-3"},
                            ui.strong(f"Claim IDs ({len(claim_ids)}): "),
                            ", ".join(str(cid) for cid in claim_ids[:30]),
                            f" ... and {len(claim_ids) - 30} more"
                            if len(claim_ids) > 30
                            else "",
                        )
                    )

            # Severity breakdown
            severity_counts = df[df["llm_flagged"]]["severity"].value_counts().to_dict()
            if severity_counts:
                content.append(ui.h5("Severity Breakdown"))
                severity_items = [
                    ui.tags.li(f"{level.title()}: {count}")
                    for level, count in severity_counts.items()
                    if level
                ]
                content.append(ui.tags.ul(*severity_items))

        if llm.summary:
            content.append(ui.h5("Analysis Summary"))

            # Split summary by batch or by sentences for better readability
            summary_text = llm.summary

            # If it's a multi-batch summary (contains newlines), show as list
            if "\n" in summary_text:
                batch_summaries = [
                    line.strip() for line in summary_text.split("\n") if line.strip()
                ]
                if len(batch_summaries) > 1:
                    content.append(
                        ui.tags.div(
                            {"class": "alert alert-info"},
                            ui.tags.ul(
                                *[ui.tags.li(summary) for summary in batch_summaries],
                                {"class": "mb-0"},
                            ),
                        )
                    )
                else:
                    content.append(ui.p(summary_text))
            else:
                # Single summary: break into sentences for readability
                sentences = [
                    s.strip() + "." for s in summary_text.split(".") if s.strip()
                ]
                if len(sentences) > 1:
                    content.append(
                        ui.tags.div(
                            {"class": "alert alert-info"},
                            ui.tags.ul(
                                *[ui.tags.li(sent) for sent in sentences],
                                {"class": "mb-0"},
                            ),
                        )
                    )
                else:
                    content.append(ui.p(summary_text))

        if llm.error:
            content.append(
                ui.div(
                    {"class": "alert alert-warning"},
                    ui.strong("Processing Note: "),
                    llm.error,
                )
            )

        if not content:
            content.append(ui.p("No additional summary available."))

        return ui.card(
            ui.card_header("Fraud Detection Summary"),
            *content,
        )

    @output
    @render.ui
    def results_table() -> ui.Tag:
        uploads = input.claims_file()
        if not uploads:
            return ui.card(
                ui.card_header("Claims Review"),
                ui.p(
                    {"class": "text-muted text-center py-5"},
                    ICON_INFO,
                    ui.br(),
                    ui.br(),
                    "Upload a CSV file to begin fraud detection analysis.",
                ),
            )

        df = consolidated()
        return ui.card(
            ui.card_header(
                ui.row(
                    ui.column(8, "Detailed Claims Review"),
                    ui.column(
                        4,
                        ui.tags.small(
                            {"class": "text-muted float-end"},
                            f"Showing {len(df)} claims",
                        ),
                    ),
                )
            ),
            ui.HTML(format_table(df)),
            full_screen=True,
        )


app_ui = ui.page_fluid(
    ui.panel_title(APP_TITLE, "Insurance Fraud Detection powered by AWS Bedrock"),
    ui.layout_columns(
        ui.card(
            ui.card_header(
                ui.row(
                    ui.column(8, ui.tags.strong("Upload Claims Data")),
                    ui.column(4, ICON_INFO),
                )
            ),
            ui.input_file(
                "claims_file",
                "Select CSV file",
                multiple=False,
                accept=[".csv"],
                button_label="Browse...",
            ),
            ui.help_text(
                "Upload a CSV file containing claim records. "
                "The LLM will analyze each claim for potential fraud "
                "indicators."
            ),
        ),
        col_widths=[12],
    ),
    ui.layout_columns(
        ui.output_ui("value_boxes"),
        col_widths=[12],
    ),
    ui.layout_columns(
        ui.output_ui("summary_card"),
        col_widths=[12],
    ),
    ui.output_ui("results_table"),
    fillable=True,
)

app = App(app_ui, server)

if __name__ == "__main__":
    app.run()
