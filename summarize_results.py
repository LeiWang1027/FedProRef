#!/usr/bin/env python3
"""
Organize FedProRef experiment results from local logs.

Outputs are written under ./results:
  - all_log_runs.csv: one row per log file
  - selected_runs.csv: one de-duplicated row per experiment name
  - summary_by_group.csv: mean/std/min/max grouped by dataset/alpha/method/variant
  - successful_summary_by_group.csv: successful non-smoke grouped statistics
  - plan_fedproref_summary.csv: the planned FedProRef sweep only
  - round_metrics.csv: per-round metrics parsed from logs
  - experiment_results.xlsx: Excel workbook for the parsed log results
  - report.md: readable Markdown report
"""

from __future__ import annotations

import csv
import math
import re
import zipfile
from collections import defaultdict
from html import escape as xml_escape
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
OUTPUT_DIR = ROOT / "results"

DATASETS = ("cifar10", "cifar100", "tinyimagenet")
ROUND_RE = re.compile(
    r"Round\s+(\d+)\s+\|\s+acc:\s*([0-9.]+)\s+\|\s+ece:\s*([0-9.]+)"
    r"\s+\|\s+nll:\s*([0-9.]+)(?:\s+\|\s+round_time:\s*([0-9.]+))?"
)
BEST_RE = re.compile(
    r"Best:\s*Round\s+(\d+)\s*\|\s*Acc=([0-9.]+)%\s*\|\s*ECE=([0-9.]+)\s*\|\s*NLL=([0-9.]+)"
)
CONFIG_RE = re.compile(r"^\s{2}([A-Za-z0-9_]+)\s*=\s*(.*?)\s*$", re.MULTILINE)
LOG_TIME_RE = re.compile(r"Experiment Log [^\d]*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
STAMP_RE = re.compile(r"_(\d{8}_\d{6})$")


@dataclass
class RoundMetric:
    log_file: str
    exp_name: str
    dataset: str
    alpha: str
    method: str
    variant: str
    round: int
    acc: float
    ece: float
    nll: float
    round_time: str = ""


@dataclass
class RunResult:
    source: str
    log_file: str
    timestamp: str = ""
    exp_name: str = ""
    dataset: str = ""
    alpha: str = ""
    method: str = ""
    variant: str = ""
    run: str = ""
    seed: str = ""
    status: str = ""
    total_rounds: int = 0
    last_round: str = ""
    final_acc: str = ""
    final_ece: str = ""
    final_nll: str = ""
    best_round: str = ""
    best_acc: str = ""
    best_ece: str = ""
    best_nll: str = ""
    comm_rounds: str = ""
    local_epochs: str = ""
    num_clients: str = ""
    select_clients: str = ""
    refiner_type: str = ""
    cal_every: str = ""
    sim_alpha: str = ""
    w_protoflow: str = ""
    note: str = ""
    config: dict[str, str] = field(default_factory=dict, repr=False)


def as_float(value: object) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_float_text(value: object) -> str:
    number = as_float(value)
    if number is None:
        return ""
    return f"{number:g}"


def infer_dataset(text: str) -> str:
    for dataset in DATASETS:
        if dataset in text:
            return dataset
    return ""


def infer_alpha(text: str) -> str:
    match = re.search(r"(?:_a|alpha)(0?\.\d+|[0-9]+(?:\.[0-9]+)?)", text)
    return normalize_float_text(match.group(1)) if match else ""


def infer_method(text: str) -> str:
    for method in ("fedavg", "proto_aug", "proto_cal", "proto_sample", "fedproref"):
        if method in text:
            return method
    return ""


def infer_run(text: str) -> str:
    match = re.search(r"_run(\d+)", text)
    return match.group(1) if match else ""


def infer_timestamp(stem: str, content: str) -> str:
    match = STAMP_RE.search(stem)
    if match:
        raw = match.group(1)
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]} {raw[9:11]}:{raw[11:13]}:{raw[13:15]}"
    match = LOG_TIME_RE.search(content)
    return match.group(1) if match else ""


def infer_variant(method: str, exp_name: str, log_file: str, config: dict[str, str]) -> str:
    text = f"{exp_name} {log_file}".lower()
    if method == "fedavg":
        return "fedavg"
    if "no_protoflow" in text or config.get("w_protoflow") == "0.0":
        return "no_protoflow"
    if "full_protoflow" in text or (
        as_float(config.get("w_protoflow")) not in (None, 0.0)
        and as_float(config.get("sim_alpha")) is not None
    ):
        return "full_protoflow"
    for name in ("proto_aug", "proto_cal", "proto_sample"):
        if name in text:
            return name
    if exp_name.startswith("plan_fedproref"):
        return "plan_fedproref"
    if exp_name.startswith(("cifar10_", "cifar100_", "tinyimagenet_")):
        return "original_repeat"
    if "smoke" in text:
        return "smoke"
    return method or "unknown"


def parse_log(path: Path) -> tuple[RunResult, list[RoundMetric]]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    config = {key: value for key, value in CONFIG_RE.findall(content)}
    stem = path.stem

    exp_name = config.get("exp_name") or stem
    dataset = config.get("dataset") or infer_dataset(stem)
    alpha = normalize_float_text(config.get("alpha") or infer_alpha(stem))
    method = config.get("method") or infer_method(stem)
    run = infer_run(exp_name) or infer_run(stem)
    timestamp = infer_timestamp(stem, content)
    variant = infer_variant(method, exp_name, path.name, config)

    rounds: list[RoundMetric] = []
    for match in ROUND_RE.finditer(content):
        rounds.append(
            RoundMetric(
                log_file=str(path.relative_to(ROOT)),
                exp_name=exp_name,
                dataset=dataset,
                alpha=alpha,
                method=method,
                variant=variant,
                round=int(match.group(1)),
                acc=float(match.group(2)),
                ece=float(match.group(3)),
                nll=float(match.group(4)),
                round_time=match.group(5) or "",
            )
        )

    best_match = None
    for best_match in BEST_RE.finditer(content):
        pass

    result = RunResult(
        source="log",
        log_file=str(path.relative_to(ROOT)),
        timestamp=timestamp,
        exp_name=exp_name,
        dataset=dataset,
        alpha=alpha,
        method=method,
        variant=variant,
        run=run,
        seed=config.get("seed", ""),
        total_rounds=len(rounds),
        comm_rounds=config.get("comm_rounds", ""),
        local_epochs=config.get("local_epochs", ""),
        num_clients=config.get("num_clients", ""),
        select_clients=config.get("select_clients", ""),
        refiner_type=config.get("refiner_type", ""),
        cal_every=config.get("cal_every", ""),
        sim_alpha=config.get("sim_alpha", ""),
        w_protoflow=config.get("w_protoflow", ""),
        config=config,
    )

    if rounds:
        final = rounds[-1]
        best_round = max(rounds, key=lambda item: item.acc)
        result.last_round = str(final.round)
        result.final_acc = f"{final.acc:.4f}"
        result.final_ece = f"{final.ece:.4f}"
        result.final_nll = f"{final.nll:.4f}"
        result.best_round = str(best_round.round)
        result.best_acc = f"{best_round.acc:.4f}"
        result.best_ece = f"{best_round.ece:.4f}"
        result.best_nll = f"{best_round.nll:.4f}"

    if best_match:
        result.best_round = best_match.group(1)
        result.best_acc = f"{float(best_match.group(2)):.4f}"
        result.best_ece = f"{float(best_match.group(3)):.4f}"
        result.best_nll = f"{float(best_match.group(4)):.4f}"

    if best_match:
        result.status = "success"
    elif "KeyboardInterrupt" in content:
        result.status = "interrupted"
        result.note = "KeyboardInterrupt"
    elif "Traceback" in content:
        result.status = "failed"
        result.note = "Traceback"
    elif rounds:
        result.status = "partial"
    else:
        result.status = "no_metrics"

    return result, rounds


def parse_logs() -> tuple[list[RunResult], list[RoundMetric]]:
    if not LOG_DIR.exists():
        return [], []

    runs: list[RunResult] = []
    rounds: list[RoundMetric] = []
    for path in sorted(LOG_DIR.glob("*.log")):
        run, run_rounds = parse_log(path)
        runs.append(run)
        rounds.extend(run_rounds)
    return runs, rounds


def run_sort_key(run: RunResult) -> tuple[str, float, str, str, int, str]:
    alpha = as_float(run.alpha)
    return (
        run.dataset,
        alpha if alpha is not None else math.inf,
        run.method,
        run.variant,
        int(run.run or 0),
        run.exp_name,
    )


def select_runs(runs: Iterable[RunResult]) -> list[RunResult]:
    by_exp: dict[str, list[RunResult]] = defaultdict(list)
    for run in runs:
        key = run.exp_name or run.log_file
        by_exp[key].append(run)

    status_rank = {
        "success": 5,
        "partial": 4,
        "interrupted": 3,
        "failed": 2,
        "no_metrics": 1,
    }

    def sort_key(run: RunResult) -> tuple[int, int, str]:
        return (
            status_rank.get(run.status, 0),
            int(run.last_round or 0),
            run.timestamp,
        )

    return sorted((max(items, key=sort_key) for items in by_exp.values()), key=run_sort_key)


def summarize(runs: Iterable[RunResult]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str, str], list[RunResult]] = defaultdict(list)
    for run in runs:
        if as_float(run.best_acc) is None:
            continue
        groups[(run.dataset, run.alpha, run.method, run.variant)].append(run)

    rows: list[dict[str, str]] = []
    for (dataset, alpha, method, variant), items in sorted(groups.items()):
        accs = [as_float(item.best_acc) for item in items]
        accs = [item for item in accs if item is not None]
        statuses = defaultdict(int)
        for item in items:
            statuses[item.status] += 1
        rows.append(
            {
                "dataset": dataset,
                "alpha": alpha,
                "method": method,
                "variant": variant,
                "count": str(len(accs)),
                "mean_best_acc": f"{mean(accs):.4f}",
                "std_best_acc": f"{pstdev(accs):.4f}" if len(accs) > 1 else "0.0000",
                "min_best_acc": f"{min(accs):.4f}",
                "max_best_acc": f"{max(accs):.4f}",
                "statuses": ";".join(f"{key}:{statuses[key]}" for key in sorted(statuses)),
            }
        )
    return rows


def write_csv(path: Path, rows: Iterable[object], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if hasattr(row, "__dataclass_fields__"):
                data = {field: getattr(row, field, "") for field in fields}
            else:
                data = {field: row.get(field, "") for field in fields}
            writer.writerow(data)


def markdown_table(rows: list[dict[str, str]], fields: list[str]) -> list[str]:
    if not rows:
        return ["_No data._"]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines


def make_report(selected: list[RunResult], summary_rows: list[dict[str, str]]) -> str:
    status_counts = defaultdict(int)
    for run in selected:
        status_counts[run.status] += 1

    plan_rows = [
        row
        for row in summary_rows
        if row["variant"] == "plan_fedproref" and row["method"] == "fedproref"
    ]
    lines = [
        "# FedProRef Experiment Data Summary",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Files",
        "",
        "- `experiment_results.xlsx`: Excel workbook for the parsed log results.",
        "- `plan_fedproref_summary.csv`: planned FedProRef sweep only.",
        "- `successful_summary_by_group.csv`: successful non-smoke grouped statistics.",
        "- `summary_by_group.csv`: grouped statistics from `selected_runs.csv`.",
        "- `selected_runs.csv`: de-duplicated experiments used for grouped summaries.",
        "- `all_log_runs.csv`: all parsed log files.",
        "- `round_metrics.csv`: per-round accuracy/ECE/NLL curves.",
        "",
        "## Selected Run Status",
        "",
    ]
    for status in sorted(status_counts):
        lines.append(f"- {status}: {status_counts[status]}")

    lines.extend(
        [
            "",
            "## Plan FedProRef Results",
            "",
            *markdown_table(
                plan_rows,
                [
                    "dataset",
                    "alpha",
                    "count",
                    "mean_best_acc",
                    "std_best_acc",
                    "min_best_acc",
                    "max_best_acc",
                    "statuses",
                ],
            ),
            "",
        ]
    )
    return "\n".join(lines)


def rows_to_lists(rows: Iterable[object], fields: list[str]) -> list[list[object]]:
    table = [fields]
    for row in rows:
        if hasattr(row, "__dataclass_fields__"):
            table.append([getattr(row, field, "") for field in fields])
        else:
            table.append([row.get(field, "") for field in fields])
    return table


def excel_column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def is_excel_number(value: object) -> bool:
    if isinstance(value, (int, float)):
        return True
    if not isinstance(value, str) or value == "":
        return False
    try:
        float(value)
    except ValueError:
        return False
    return True


def sheet_xml(rows: list[list[object]]) -> str:
    xml_rows = []
    for row_index, row in enumerate(rows, start=1):
        cells = []
        for col_index, value in enumerate(row, start=1):
            ref = f"{excel_column_name(col_index)}{row_index}"
            if is_excel_number(value):
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                text = xml_escape(str(value), quote=False)
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>')
        xml_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
        'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        f'<sheetData>{"".join(xml_rows)}</sheetData></worksheet>'
    )


def write_xlsx(path: Path, sheets: list[tuple[str, list[list[object]]]]) -> None:
    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for index, _ in enumerate(sheets, start=1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append('</Types>')

    workbook_sheets = []
    workbook_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for index, (name, _) in enumerate(sheets, start=1):
        safe_name = xml_escape(name[:31])
        workbook_sheets.append(f'<sheet name="{safe_name}" sheetId="{index}" r:id="rId{index}"/>')
        workbook_rels.append(
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
        )
    style_rel_id = len(sheets) + 1
    workbook_rels.append(
        f'<Relationship Id="rId{style_rel_id}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    )
    workbook_rels.append('</Relationships>')

    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{"".join(workbook_sheets)}</sheets></workbook>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        '</styleSheet>'
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "".join(content_types))
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", "".join(workbook_rels))
        archive.writestr("xl/styles.xml", styles)
        for index, (_, rows) in enumerate(sheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", sheet_xml(rows))



def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    all_log_runs, round_metrics = parse_logs()
    selected_runs = select_runs(all_log_runs)
    summary_rows = summarize(selected_runs)
    successful_summary_rows = summarize(
        run for run in selected_runs if run.status == "success" and run.variant != "smoke"
    )
    plan_summary_rows = [
        row
        for row in summary_rows
        if row["variant"] == "plan_fedproref" and row["method"] == "fedproref"
    ]
    run_fields = [
        "source",
        "log_file",
        "timestamp",
        "exp_name",
        "dataset",
        "alpha",
        "method",
        "variant",
        "run",
        "seed",
        "status",
        "total_rounds",
        "last_round",
        "final_acc",
        "final_ece",
        "final_nll",
        "best_round",
        "best_acc",
        "best_ece",
        "best_nll",
        "comm_rounds",
        "local_epochs",
        "num_clients",
        "select_clients",
        "refiner_type",
        "cal_every",
        "sim_alpha",
        "w_protoflow",
        "note",
    ]
    round_fields = [
        "log_file",
        "exp_name",
        "dataset",
        "alpha",
        "method",
        "variant",
        "round",
        "acc",
        "ece",
        "nll",
        "round_time",
    ]
    summary_fields = [
        "dataset",
        "alpha",
        "method",
        "variant",
        "count",
        "mean_best_acc",
        "std_best_acc",
        "min_best_acc",
        "max_best_acc",
        "statuses",
    ]

    write_csv(OUTPUT_DIR / "all_log_runs.csv", sorted(all_log_runs, key=run_sort_key), run_fields)
    write_csv(OUTPUT_DIR / "selected_runs.csv", selected_runs, run_fields)
    write_csv(OUTPUT_DIR / "summary_by_group.csv", summary_rows, summary_fields)
    write_csv(OUTPUT_DIR / "successful_summary_by_group.csv", successful_summary_rows, summary_fields)
    write_csv(OUTPUT_DIR / "plan_fedproref_summary.csv", plan_summary_rows, summary_fields)
    write_csv(OUTPUT_DIR / "round_metrics.csv", round_metrics, round_fields)
    for stale_file in ("legacy_experiment_results.csv", "legacy_summary.csv"):
        stale_path = OUTPUT_DIR / stale_file
        if stale_path.exists():
            stale_path.unlink()

    write_xlsx(
        OUTPUT_DIR / "experiment_results.xlsx",
        [
            ("Plan FedProRef", rows_to_lists(plan_summary_rows, summary_fields)),
            ("Successful Summary", rows_to_lists(successful_summary_rows, summary_fields)),
            ("Selected Runs", rows_to_lists(selected_runs, run_fields)),
            ("All Log Runs", rows_to_lists(sorted(all_log_runs, key=run_sort_key), run_fields)),
            ("Round Metrics", rows_to_lists(round_metrics, round_fields)),
        ],
    )

    report = make_report(selected_runs, summary_rows)
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")

    print(f"Parsed log files: {len(all_log_runs)}")
    print(f"Selected runs:    {len(selected_runs)}")
    print(f"Round records:    {len(round_metrics)}")
    print(f"Output directory: {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
