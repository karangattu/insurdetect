import pandas as pd
from shiny_app.chat_utils import create_chat
from great_tables import GT
from shiny import reactive
from shiny.express import input, render, ui


SAMPLE_INPUT = """Claim C-2001 for $28,000 filed on Feb 1, 2025 (policy P-001 started Dec 15, 2024).
Auto claim: Rear-end collision; police report filed; staged accident suspected."""


ui.page_opts(title="insurdetect, a fraud detection app", fillable=True)

with ui.layout_sidebar():
    with ui.sidebar(width=400):
        ui.h4("Analyze Claims")

        ui.input_radio_buttons(
            "input_mode",
            "Input Method:",
            {"single": "Single Claim", "batch": "Batch Upload (CSV)"},
            selected="single"
        )

        @render.ui
        def input_section():
            if input.input_mode() == "single":
                return ui.TagList(
                    ui.input_text_area(
                        "claims_input",
                        "Claim Description:",
                        value=SAMPLE_INPUT,
                        rows=8,
                        width="100%",
                    ),
                    ui.input_action_button("analyze", "Analyze Claim", class_="btn-primary w-100")
                )
            else:
                return ui.TagList(
                    ui.input_file(
                        "claims_file",
                        "Upload CSV file:",
                        accept=[".csv"],
                        placeholder="Select a CSV file with 'input' column"
                    ),
                    ui.input_action_button("analyze_batch", "Analyze All Claims", class_="btn-primary w-100 mt-2")
                )

    with ui.card():
        ui.card_header("Fraud Analysis Results")

        @render.ui
        @reactive.event(input.analyze)
        async def analysis_result():
            claim_text = input.claims_input().strip()
            if not claim_text:
                return ui.p(
                    "Please enter a claim description.", class_="text-warning p-3"
                )

            chat = create_chat()

            response = await chat.chat_async(
                f"Analyze this insurance claim:\n{claim_text}", stream=False
            )
            result_text = await response.get_content()

            upper_text = result_text.upper()
            risk_level, severity_class = next(
                (
                    (level, cls)
                    for level, cls in [
                        ("HIGH", "bg-danger"),
                        ("MEDIUM", "bg-warning text-dark"),
                        ("LOW", "bg-info"),
                        ("NONE", "bg-success"),
                    ]
                    if level in upper_text
                ),
                ("UNKNOWN", "bg-secondary"),
            )

            payment_decision, payment_class = next(
                (
                    (dec, cls)
                    for dec, cls in [
                        ("PENDING", "bg-warning text-dark"),
                        ("AUTO-APPROVE", "bg-success"),
                    ]
                    if dec in upper_text
                ),
                ("UNKNOWN", "bg-secondary"),
            )

            df = pd.DataFrame(
                {
                    "Field": ["Risk Level", "Decision", "Analysis"],
                    "Value": [
                        f'<span class="badge {severity_class}">{risk_level}</span>',
                        f'<span class="badge {payment_class}">{payment_decision}</span>',
                        result_text,
                    ],
                }
            )

            gt_table = (
                GT(df)
                .fmt_markdown(columns="Value")
                .tab_options(
                    table_font_size="14px",
                    heading_background_color="#f8f9fa",
                    column_labels_background_color="#e9ecef",
                )
            )

            return ui.HTML(gt_table.as_raw_html())

        @render.ui
        @reactive.event(input.analyze_batch)
        async def batch_analysis_result():
            file_info = input.claims_file()
            if not file_info:
                return ui.p("Please upload a CSV file.", class_="text-warning p-3")

            try:
                # Read the CSV file
                csv_content = file_info[0]["datapath"]
                df_claims = pd.read_csv(csv_content)

                if "input" not in df_claims.columns:
                    return ui.p(
                        "CSV file must contain an 'input' column with claim descriptions.",
                        class_="text-danger p-3"
                    )

                # Process each claim
                results = []

                for i, (_, row) in enumerate(df_claims.iterrows()):
                    claim_text = str(row["input"]).strip()
                    if not claim_text or claim_text == "nan":
                        continue

                    chat = create_chat()
                    response = await chat.chat_async(
                        f"Analyze this insurance claim:\n{claim_text}", stream=False
                    )
                    result_text = await response.get_content()

                    upper_text = result_text.upper()

                    # Extract risk level
                    risk_level = next(
                        (level for level in ["HIGH", "MEDIUM", "LOW", "NONE"] if level in upper_text),
                        "UNKNOWN"
                    )

                    # Extract payment decision
                    payment_decision = next(
                        (dec for dec in ["PENDING", "AUTO-APPROVE"] if dec in upper_text),
                        "UNKNOWN"
                    )

                    # Extract claim ID from the text if possible
                    claim_id = f"Claim {i + 1}"
                    if "Claim C-" in claim_text:
                        claim_id = claim_text.split()[0] + " " + claim_text.split()[1]

                    results.append({
                        "Claim ID": claim_id,
                        "Risk Level": risk_level,
                        "Decision": payment_decision,
                        "Claim Description": claim_text[:100] + "..." if len(claim_text) > 100 else claim_text,
                        "Full Analysis": result_text
                    })

                # Sort by risk level priority
                risk_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NONE": 3, "UNKNOWN": 4}
                results_df = pd.DataFrame(results)
                results_df["_sort_order"] = results_df["Risk Level"].map(risk_order)
                results_df = results_df.sort_values("_sort_order").drop("_sort_order", axis=1)

                # Create styled display dataframe
                display_df = results_df.copy()

                # Apply badges to Risk Level
                display_df["Risk Level"] = display_df["Risk Level"].apply(
                    lambda x: {
                        "HIGH": f'<span class="badge bg-danger">{x}</span>',
                        "MEDIUM": f'<span class="badge bg-warning text-dark">{x}</span>',
                        "LOW": f'<span class="badge bg-info">{x}</span>',
                        "NONE": f'<span class="badge bg-success">{x}</span>',
                        "UNKNOWN": f'<span class="badge bg-secondary">{x}</span>',
                    }.get(x, x)
                )

                # Apply badges to Decision
                display_df["Decision"] = display_df["Decision"].apply(
                    lambda x: {
                        "PENDING": f'<span class="badge bg-warning text-dark">{x}</span>',
                        "AUTO-APPROVE": f'<span class="badge bg-success">{x}</span>',
                        "UNKNOWN": f'<span class="badge bg-secondary">{x}</span>',
                    }.get(x, x)
                )

                # Create summary stats
                risk_counts = results_df["Risk Level"].value_counts().to_dict()
                summary_html = f"""
                <div class="alert alert-info mb-3">
                    <h5>Analysis Summary</h5>
                    <p><strong>Total Claims Analyzed:</strong> {len(results_df)}</p>
                    <p><strong>Risk Distribution:</strong></p>
                    <ul>
                        <li>HIGH: {risk_counts.get('HIGH', 0)}</li>
                        <li>MEDIUM: {risk_counts.get('MEDIUM', 0)}</li>
                        <li>LOW: {risk_counts.get('LOW', 0)}</li>
                        <li>NONE: {risk_counts.get('NONE', 0)}</li>
                    </ul>
                </div>
                """

                # Create the table
                table_df = display_df[["Claim ID", "Risk Level", "Decision", "Claim Description"]].copy()

                gt_table = (
                    GT(table_df)
                    .fmt_markdown(columns=["Risk Level", "Decision"])
                    .tab_options(
                        table_font_size="13px",
                        heading_background_color="#f8f9fa",
                        column_labels_background_color="#e9ecef",
                        table_width="100%"
                    )
                )

                return ui.HTML(summary_html + gt_table.as_raw_html())

            except Exception as e:
                return ui.p(
                    f"Error processing file: {str(e)}",
                    class_="text-danger p-3"
                )
