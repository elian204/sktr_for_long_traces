#!/usr/bin/env python3
"""Build a focused DiffAct/SKTR findings report from validated artifacts."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    FrameBreak,
    KeepTogether,
    ListFlowable,
    ListItem,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


RUN_ROOT = Path("/data1/eli-bogdanov/sktr_runs")
OUT_BASE = RUN_ROOT / "DIFFACT_SKTR_FINDINGS_FOCUSED_REPORT"

TAX_DIR = RUN_ROOT / "diffact_error_taxonomy_v1"
METRIC_TABLES = {
    "50Salads": RUN_ROOT
    / "diffact_50salads_allfolds_resumable_6ba8868_chunk11"
    / "table_diffact_all_folds.csv",
    "Breakfast": RUN_ROOT
    / "diffact_breakfast_unique199_f14fd99_chunk11_w10"
    / "table_diffact_all_folds.csv",
    "GTEA": RUN_ROOT
    / "diffact_gtea_allfolds_resumable_6ba8868_chunk11_w7"
    / "table_diffact_all_folds.csv",
}
CEILING_TABLES = {
    "GTEA": [
        RUN_ROOT / "sktr_ceiling_analysis_gtea_skip_all_v4" / "all_ceiling_cases.csv"
    ],
    "Breakfast": [
        RUN_ROOT / "sktr_ceiling_breakfast_skip_all_v1" / "all_ceiling_cases.csv"
    ],
    "50Salads": [
        RUN_ROOT
        / "sktr_ceiling_50salads_fold1_run_rerun_64gb_v1"
        / "completed_ceiling_cases.csv",
        RUN_ROOT
        / "sktr_ceiling_50salads_isolated_v1"
        / "completed_ceiling_cases.csv",
    ],
}


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def pp(value: float) -> str:
    return f"{value:+.2f} pp"


def num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def ratio(count: int, denom: int, *, lower_bound: bool = False) -> str:
    prefix = ">=" if lower_bound else ""
    return f"{prefix}{count}/{denom} ({prefix}{count / denom * 100:.0f}%)"


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("\n", " ")

    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(out)


def load_metrics() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for dataset, path in METRIC_TABLES.items():
        df = pd.read_csv(path)
        for method, group in df.groupby("Method"):
            means = group[["Edit", "F1@10", "F1@25", "F1@50", "Acc"]].mean()
            row = {"Dataset": dataset, "Method": method}
            row.update({k: float(v) for k, v in means.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def metrics_rows(metrics: pd.DataFrame) -> List[List[str]]:
    rows: List[List[str]] = []
    for dataset in ["50Salads", "GTEA", "Breakfast"]:
        sub = metrics[metrics["Dataset"] == dataset]
        for method in ["DiffAct (argmax)", "DiffAct + SKTR"]:
            row = sub[sub["Method"] == method].iloc[0]
            label = "Argmax" if "argmax" in method else "SKTR"
            rows.append(
                [
                    dataset,
                    label,
                    num(row["Edit"], 2),
                    num(row["F1@10"], 2),
                    num(row["F1@25"], 2),
                    num(row["F1@50"], 2),
                    num(row["Acc"], 2),
                ]
            )
    return rows


def metric_delta_rows(metrics: pd.DataFrame) -> List[List[str]]:
    rows: List[List[str]] = []
    for dataset in ["50Salads", "GTEA", "Breakfast"]:
        sub = metrics[metrics["Dataset"] == dataset]
        arg = sub[sub["Method"] == "DiffAct (argmax)"].iloc[0]
        sktr = sub[sub["Method"] == "DiffAct + SKTR"].iloc[0]
        rows.append(
            [
                dataset,
                pp(float(sktr["Acc"] - arg["Acc"])),
                pp(float(sktr["Edit"] - arg["Edit"])),
                pp(float(sktr["F1@10"] - arg["F1@10"])),
                pp(float(sktr["F1@25"] - arg["F1@25"])),
                pp(float(sktr["F1@50"] - arg["F1@50"])),
            ]
        )
    return rows


def load_ceiling() -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for dataset, paths in CEILING_TABLES.items():
        out[dataset] = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return out


def ceiling_rows(ceiling: Dict[str, pd.DataFrame]) -> List[List[str]]:
    rows: List[List[str]] = []
    for dataset in ["GTEA", "Breakfast", "50Salads"]:
        df = ceiling[dataset]
        n = len(df)
        gt_prefix = int(df["gt_accepted_exact"].astype(bool).sum())
        gt_tau = int(df["gt_accepted_exact_tau_completed"].astype(bool).sum())
        argmax_no_order = int(
            (
                df["argmax_log_moves"].astype(int)
                + df["argmax_model_moves"].astype(int)
            ).eq(0).sum()
        )
        note = "full"
        prefix_lower = False
        tau_lower = False
        if dataset == "Breakfast":
            note = "run chunking; lower bound"
            prefix_lower = True
            tau_lower = True
        elif dataset == "50Salads":
            note = "run chunking; tau robust"
        rows.append(
            [
                dataset,
                str(n),
                ratio(gt_prefix, n, lower_bound=prefix_lower),
                ratio(gt_tau, n, lower_bound=tau_lower),
                ratio(argmax_no_order, n),
                note,
            ]
        )
    return rows


def load_taxonomy() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_dataset = pd.read_csv(TAX_DIR / "error_taxonomy_per_dataset.csv")
    outliers = pd.read_csv(TAX_DIR / "error_taxonomy_outliers.csv")
    per_case = pd.read_csv(TAX_DIR / "error_taxonomy_per_case.csv")
    return per_dataset, outliers, per_case


def taxonomy_rows(per_dataset: pd.DataFrame) -> List[List[str]]:
    rows: List[List[str]] = []
    for dataset in ["GTEA", "Breakfast", "50Salads"]:
        row = per_dataset[
            (per_dataset["dataset"].str.lower() == dataset.lower())
            & (per_dataset["system"] == "argmax")
        ].iloc[0]
        rows.append(
            [
                dataset,
                str(int(row["total_errors"])),
                pct(float(row["boundary_w10_share"])),
                pct(float(row["boundary_w25_share"])),
                pct(float(row["boundary_w50_share"])),
                pct(float(row["long_substitution_share"])),
                pct(float(row["class_confusion_share"])),
                pct(float(row["residual_share"])),
            ]
        )
    return rows


def shift_rows(per_dataset: pd.DataFrame) -> List[List[str]]:
    rows: List[List[str]] = []
    for dataset in ["GTEA", "Breakfast", "50Salads"]:
        sub = per_dataset[per_dataset["dataset"].str.lower() == dataset.lower()]
        arg = sub[sub["system"] == "argmax"].iloc[0]
        sktr = sub[sub["system"] == "sktr"].iloc[0]
        total_delta = int(sktr["total_errors"] - arg["total_errors"])
        rows.append(
            [
                dataset,
                f"{total_delta:+d}",
                f"{int(sktr['boundary_w25'] - arg['boundary_w25']):+d}",
                f"{int(sktr['long_substitution'] - arg['long_substitution']):+d}",
                f"{int(sktr['class_confusion'] - arg['class_confusion']):+d}",
                f"{int(sktr['residual'] - arg['residual']):+d}",
            ]
        )
    return rows


def outlier_rows(outliers: pd.DataFrame) -> List[List[str]]:
    rows: List[List[str]] = []
    targets = [
        ("50salads", 1, 1),
        ("50salads", 5, 49),
        ("gtea", 2, 7),
        ("gtea", 2, 8),
        ("gtea", 2, 9),
        ("gtea", 2, 11),
        ("gtea", 2, 12),
        ("gtea", 2, 13),
    ]
    for dataset, fold, case in targets:
        for system in ["argmax", "sktr"]:
            row = outliers[
                (outliers["dataset"].str.lower() == dataset)
                & (outliers["fold"] == fold)
                & (outliers["case_id"].astype(str) == str(case))
                & (outliers["system"] == system)
            ].iloc[0]
            rows.append(
                [
                    dataset,
                    str(fold),
                    str(case),
                    system,
                    str(int(row["total_errors"])),
                    pct(float(row["boundary_w25_share"])),
                    pct(float(row["long_substitution_share"])),
                    pct(float(row["class_confusion_share"])),
                    pct(float(row["residual_share"])),
                ]
            )
    return rows


def top_findings() -> List[str]:
    return [
        "DiffAct residual errors are not primarily order errors. The recovered activity order usually already fits the discovered structure.",
        "GTEA is boundary-heavy: 68.1% of argmax errors are within 25 frames of a GT boundary and match an adjacent GT activity.",
        "Breakfast and 50Salads are long-span-heavy: their largest argmax error class is long_substitution, not boundary jitter.",
        "SKTR does not create a stable aggregate win. It is close to neutral on 50Salads and negative on GTEA and Breakfast.",
        "On 50Salads, the visible SKTR effect is rare high-variance long substitutions: one catastrophic harm case and one large help case dominate the fold-level story.",
    ]


def definitions() -> List[Tuple[str, str]]:
    return [
        (
            "boundary_w25",
            "Primary error bucket. A wrong frame within 25 frames of a GT segment boundary where the predicted class is one of the two adjacent GT segment labels. Interpreted as the right activity with the wrong edge timing.",
        ),
        (
            "boundary_w10 / boundary_w50",
            "Width-sensitivity counts for the same boundary rule using 10 or 50 frames. These are reported in parallel and are not part of the five mutually-exclusive category sum.",
        ),
        (
            "over_segmentation_island",
            "A short wrong-class span of length <=25 frames embedded inside a single otherwise correctly predicted GT segment.",
        ),
        (
            "long_substitution",
            "A contiguous wrong-class span of length >=100 frames where >=90% of the span is one wrong predicted class. This captures extended commitment to a single wrong class.",
        ),
        (
            "class_confusion",
            "A wrong frame whose predicted class is in the top-3 classes most confused with that GT class on the fold's training cases only.",
        ),
        (
            "residual",
            "Any wrong frame not assigned to the earlier buckets.",
        ),
        (
            "gt_accepted_exact",
            "Case-level ceiling flag: the GT collapsed activity sequence replays through the discovered net with zero log/model moves. This means the order is representable.",
        ),
        (
            "gt_accepted_exact_tau_completed",
            "Case-level ceiling flag: GT is prefix-exact and the resulting marking can reach the final marking using only tau transitions. This means the net is a complete accepting model for the case.",
        ),
        (
            "argmax_log_moves / argmax_model_moves",
            "Case-level order-deviation counts for the collapsed argmax sequence. Zero plus zero means argmax has no order violation under the discovered net.",
        ),
        (
            "sum_check_ok",
            "Sanity flag: the five primary category counts sum exactly to total_errors for that case/system row.",
        ),
    ]


def build_markdown(
    *,
    metrics: pd.DataFrame,
    per_dataset: pd.DataFrame,
    outliers: pd.DataFrame,
    ceiling: Dict[str, pd.DataFrame],
) -> str:
    lines: List[str] = []
    lines.extend(
        [
            "# DiffAct + SKTR Focused Findings Report",
            "",
            "This report is a findings-first companion to the full investigation summary.",
            "It uses only validated, existing artifacts: completed DiffAct/SKTR case CSVs,",
            "validated ceiling CSVs, and the finalized error-taxonomy outputs. No SKTR",
            "reruns, Petri-net rediscovery, or ceiling recomputation were performed.",
            "",
            "## Main Findings",
            "",
        ]
    )
    for item in top_findings():
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Aggregate TAS Metrics",
            "",
            md_table(
                ["Dataset", "System", "Edit", "F1@10", "F1@25", "F1@50", "Acc"],
                metrics_rows(metrics),
            ),
            "",
            "SKTR minus argmax:",
            "",
            md_table(
                ["Dataset", "Acc", "Edit", "F1@10", "F1@25", "F1@50"],
                metric_delta_rows(metrics),
            ),
            "",
            "## Error-Type Definitions and Columns",
            "",
        ]
    )
    lines.append(md_table(["Column", "Meaning"], definitions()))
    lines.extend(
        [
            "",
            "The five primary taxonomy buckets are priority-ordered and mutually exclusive:",
            "`boundary_w25`, `over_segmentation_island`, `long_substitution`,",
            "`class_confusion`, and `residual`. For every case/system row, these",
            "five counts sum exactly to `total_errors`.",
            "",
            "## DiffAct Argmax Error Breakdown",
            "",
            md_table(
                [
                    "Dataset",
                    "Total errors",
                    "boundary_w10",
                    "boundary_w25",
                    "boundary_w50",
                    "long_substitution",
                    "class_confusion",
                    "residual",
                ],
                taxonomy_rows(per_dataset),
            ),
            "",
            "## How SKTR Shifts the Error Distribution",
            "",
            md_table(
                [
                    "Dataset",
                    "Delta total errors",
                    "Delta boundary_w25",
                    "Delta long_substitution",
                    "Delta class_confusion",
                    "Delta residual",
                ],
                shift_rows(per_dataset),
            ),
            "",
            "## Case-Level Order Context",
            "",
            md_table(
                [
                    "Dataset",
                    "Cases",
                    "GT prefix-exact",
                    "GT tau-completed",
                    "Argmax no order violation",
                    "Measurement note",
                ],
                ceiling_rows(ceiling),
            ),
            "",
            "## Named Outlier Breakdown",
            "",
            md_table(
                [
                    "Dataset",
                    "Fold",
                    "Case",
                    "System",
                    "Total errors",
                    "boundary_w25",
                    "long_substitution",
                    "class_confusion",
                    "residual",
                ],
                outlier_rows(outliers),
            ),
            "",
            "## Interpretation Boundaries",
            "",
            "- This report is descriptive. It does not propose or evaluate new decoders.",
            "- Breakfast and 50Salads ceiling numbers use run chunking where noted; those values should be read with the caveats shown in the table.",
            "- `class_confusion` is fold-pure: the top-3 confusion lists are built from training-case DiffAct argmax versus training GT only.",
            "- Width sensitivity is explicit: `boundary_w10`, `boundary_w25`, and `boundary_w50` are all reported.",
            "",
            "## Source Artifacts",
            "",
            f"- Error taxonomy: `{TAX_DIR}`",
            "- Metrics tables:",
        ]
    )
    for dataset, path in METRIC_TABLES.items():
        lines.append(f"  - {dataset}: `{path}`")
    lines.append("- Ceiling artifacts:")
    for dataset, paths in CEILING_TABLES.items():
        lines.append(f"  - {dataset}: " + ", ".join(f"`{p}`" for p in paths))
    return "\n".join(lines) + "\n"


def build_html(markdown_text: str) -> str:
    # Small markdown subset renderer for the generated report.
    lines = markdown_text.splitlines()
    body: List[str] = []
    in_ul = False
    in_table = False
    table_headers: List[str] = []
    table_rows: List[List[str]] = []

    def flush_ul() -> None:
        nonlocal in_ul
        if in_ul:
            body.append("</ul>")
            in_ul = False

    def flush_table() -> None:
        nonlocal in_table, table_headers, table_rows
        if not in_table:
            return
        body.append("<table>")
        body.append("<thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in table_headers) + "</tr></thead>")
        body.append("<tbody>")
        for row in table_rows:
            body.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>")
        body.append("</tbody></table>")
        in_table = False
        table_headers = []
        table_rows = []

    def parse_table_line(line: str) -> List[str]:
        return [part.strip() for part in line.strip().strip("|").split("|")]

    for line in lines:
        if line.startswith("| "):
            flush_ul()
            parts = parse_table_line(line)
            if not in_table:
                in_table = True
                table_headers = parts
                table_rows = []
            elif all(set(part) <= {"-"} for part in parts):
                pass
            else:
                table_rows.append(parts)
            continue
        flush_table()
        if line.startswith("- "):
            if not in_ul:
                body.append("<ul>")
                in_ul = True
            body.append(f"<li>{html.escape(line[2:])}</li>")
        elif line.startswith("# "):
            flush_ul()
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            flush_ul()
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif not line.strip():
            flush_ul()
        else:
            flush_ul()
            body.append(f"<p>{html.escape(line)}</p>")
    flush_table()
    flush_ul()

    css = """
    body { font-family: Arial, sans-serif; margin: 40px; color: #1f2933; }
    h1 { color: #111827; font-size: 28px; margin-bottom: 6px; }
    h2 { color: #1f2937; font-size: 20px; margin-top: 28px; border-bottom: 1px solid #d0d7de; padding-bottom: 4px; }
    p, li { font-size: 13px; line-height: 1.45; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0 20px; font-size: 11px; }
    th { background: #e5eef7; color: #111827; text-align: left; }
    th, td { border: 1px solid #c8d1dc; padding: 6px 7px; vertical-align: top; }
    tr:nth-child(even) td { background: #f8fafc; }
    code { background: #f3f4f6; padding: 1px 3px; border-radius: 3px; }
    """
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{''.join(body)}</body></html>"


def pdf_styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#111827"),
            spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=colors.HexColor("#1f2937"),
            spaceBefore=8,
            spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12,
            textColor=colors.HexColor("#1f2933"),
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.4,
            leading=9,
            textColor=colors.HexColor("#1f2933"),
        ),
        "cell": ParagraphStyle(
            "cell",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.2,
            leading=8.5,
            textColor=colors.HexColor("#1f2933"),
        ),
        "cell_center": ParagraphStyle(
            "cell_center",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.2,
            leading=8.5,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1f2933"),
        ),
        "th": ParagraphStyle(
            "th",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.4,
            leading=8.8,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#111827"),
        ),
    }


def make_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    styles: Dict[str, ParagraphStyle],
    widths: Sequence[float] | None = None,
    center_from: int = 1,
) -> Table:
    data: List[List[Any]] = [
        [Paragraph(str(h), styles["th"]) for h in headers]
    ]
    for row in rows:
        cells = []
        for idx, cell in enumerate(row):
            style = styles["cell"] if idx < center_from else styles["cell_center"]
            cells.append(Paragraph(str(cell), style))
        data.append(cells)
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5eef7")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c8d1dc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def page_footer(canvas: Any, doc: BaseDocTemplate) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawString(1.4 * cm, 0.9 * cm, "DiffAct + SKTR focused findings report")
    canvas.drawRightString(
        doc.pagesize[0] - 1.4 * cm,
        0.9 * cm,
        f"Page {doc.page}",
    )
    canvas.restoreState()


def build_pdf(
    *,
    path: Path,
    metrics: pd.DataFrame,
    per_dataset: pd.DataFrame,
    outliers: pd.DataFrame,
    ceiling: Dict[str, pd.DataFrame],
) -> None:
    styles = pdf_styles()
    portrait_frame = Frame(
        1.35 * cm,
        1.4 * cm,
        A4[0] - 2.7 * cm,
        A4[1] - 2.7 * cm,
        id="portrait",
    )
    landscape_frame = Frame(
        1.25 * cm,
        1.25 * cm,
        landscape(A4)[0] - 2.5 * cm,
        landscape(A4)[1] - 2.5 * cm,
        id="landscape",
    )
    doc = BaseDocTemplate(str(path))
    doc.addPageTemplates(
        [
            PageTemplate(id="portrait", pagesize=A4, frames=[portrait_frame], onPage=page_footer),
            PageTemplate(id="landscape", pagesize=landscape(A4), frames=[landscape_frame], onPage=page_footer),
        ]
    )

    story: List[Any] = []
    story.append(Paragraph("DiffAct + SKTR Focused Findings Report", styles["title"]))
    story.append(
        Paragraph(
            "A compact, table-first report of the validated error characterization. "
            "All numbers come from existing artifacts; no SKTR reruns, Petri-net rediscovery, "
            "or ceiling recomputation were performed.",
            styles["body"],
        )
    )
    story.append(Paragraph("Main Findings", styles["h1"]))
    story.append(
        ListFlowable(
            [ListItem(Paragraph(item, styles["body"])) for item in top_findings()],
            bulletType="bullet",
            leftIndent=14,
        )
    )
    story.append(Paragraph("Aggregate TAS Metrics", styles["h1"]))
    story.append(
        make_table(
            ["Dataset", "System", "Edit", "F1@10", "F1@25", "F1@50", "Acc"],
            metrics_rows(metrics),
            styles=styles,
            widths=[2.6 * cm, 2.4 * cm, 1.6 * cm, 1.6 * cm, 1.6 * cm, 1.6 * cm, 1.6 * cm],
        )
    )
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        make_table(
            ["Dataset", "Acc delta", "Edit delta", "F1@10 delta", "F1@25 delta", "F1@50 delta"],
            metric_delta_rows(metrics),
            styles=styles,
            widths=[2.7 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("Error Types and Output Columns", styles["h1"]))
    story.append(
        Paragraph(
            "The five primary buckets are priority-ordered and mutually exclusive. "
            "For each case/system row, boundary_w25 + over_segmentation_island + "
            "long_substitution + class_confusion + residual equals total_errors.",
            styles["body"],
        )
    )
    story.append(
        make_table(
            ["Column", "Meaning"],
            definitions(),
            styles=styles,
            widths=[4.6 * cm, 13.0 * cm],
            center_from=99,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("DiffAct Argmax Error Breakdown", styles["h1"]))
    story.append(
        make_table(
            [
                "Dataset",
                "Total errors",
                "boundary_w10",
                "boundary_w25",
                "boundary_w50",
                "long_substitution",
                "class_confusion",
                "residual",
            ],
            taxonomy_rows(per_dataset),
            styles=styles,
            widths=[2.4 * cm, 2.1 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 2.4 * cm, 2.2 * cm, 2.0 * cm],
        )
    )
    story.append(Paragraph("How SKTR shifts the distribution", styles["h1"]))
    story.append(
        make_table(
            [
                "Dataset",
                "Delta total errors",
                "Delta boundary_w25",
                "Delta long_substitution",
                "Delta class_confusion",
                "Delta residual",
            ],
            shift_rows(per_dataset),
            styles=styles,
            widths=[2.6 * cm, 2.5 * cm, 3.0 * cm, 3.2 * cm, 3.0 * cm, 2.5 * cm],
        )
    )
    story.append(Paragraph("Case-level order context", styles["h1"]))
    story.append(
        make_table(
            [
                "Dataset",
                "Cases",
                "GT prefix-exact",
                "GT tau-completed",
                "Argmax no order violation",
                "Measurement note",
            ],
            ceiling_rows(ceiling),
            styles=styles,
            widths=[2.4 * cm, 1.3 * cm, 3.1 * cm, 3.1 * cm, 3.6 * cm, 4.0 * cm],
        )
    )

    story.append(NextPageTemplate("landscape"))
    story.append(PageBreak())
    story.append(Paragraph("Named Outlier Breakdown", styles["h1"]))
    story.append(
        Paragraph(
            "These are the cases explicitly named in the investigation: GTEA fold-2 "
            "non-tau-completed cases and the two 50Salads high-magnitude cases.",
            styles["body"],
        )
    )
    story.append(
        make_table(
            [
                "Dataset",
                "Fold",
                "Case",
                "System",
                "Total errors",
                "boundary_w25",
                "long_substitution",
                "class_confusion",
                "residual",
            ],
            outlier_rows(outliers),
            styles=styles,
            widths=[2.4 * cm, 1.2 * cm, 1.3 * cm, 1.7 * cm, 2.0 * cm, 2.4 * cm, 2.8 * cm, 2.6 * cm, 2.0 * cm],
        )
    )

    story.append(NextPageTemplate("portrait"))
    story.append(PageBreak())
    story.append(Paragraph("Interpretation Boundaries", styles["h1"]))
    boundaries = [
        "This report is descriptive. It does not propose or evaluate decoder changes.",
        "Breakfast and 50Salads ceiling values use run chunking where noted; the table states the measurement basis.",
        "Class-confusion lists are fold-pure and built from training-case DiffAct argmax versus training GT only.",
        "Manual validation and all sanity flags are in the canonical taxonomy artifact directory.",
    ]
    story.append(
        ListFlowable(
            [ListItem(Paragraph(item, styles["body"])) for item in boundaries],
            bulletType="bullet",
            leftIndent=14,
        )
    )
    story.append(Paragraph("Source Artifacts", styles["h1"]))
    source_rows: List[List[str]] = [["Error taxonomy", str(TAX_DIR)]]
    for dataset, path_item in METRIC_TABLES.items():
        source_rows.append([f"{dataset} metrics", str(path_item)])
    for dataset, paths in CEILING_TABLES.items():
        source_rows.append([f"{dataset} ceiling", "\n".join(str(p) for p in paths)])
    story.append(make_table(["Artifact", "Path"], source_rows, styles=styles, widths=[4.2 * cm, 13.4 * cm], center_from=99))

    doc.build(story)


def main() -> None:
    metrics = load_metrics()
    per_dataset, outliers, _ = load_taxonomy()
    ceiling = load_ceiling()

    markdown = build_markdown(
        metrics=metrics,
        per_dataset=per_dataset,
        outliers=outliers,
        ceiling=ceiling,
    )
    md_path = OUT_BASE.with_suffix(".md")
    html_path = OUT_BASE.with_suffix(".html")
    pdf_path = OUT_BASE.with_suffix(".pdf")
    md_path.write_text(markdown)
    html_path.write_text(build_html(markdown))
    build_pdf(
        path=pdf_path,
        metrics=metrics,
        per_dataset=per_dataset,
        outliers=outliers,
        ceiling=ceiling,
    )
    print(f"Wrote {md_path}")
    print(f"Wrote {html_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
