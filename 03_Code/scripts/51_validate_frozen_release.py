#!/usr/bin/env python3
"""Validate the frozen RAISE-IDS release without regenerating or overwriting outputs."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json
import math

SCRIPT_PATH = Path(__file__).resolve()
ROOT = next(
    parent for parent in (SCRIPT_PATH.parents[1], SCRIPT_PATH.parents[2], SCRIPT_PATH.parents[3])
    if (parent / "results_summary").exists()
)
RESULTS = ROOT / "results_summary"
TOL = 5e-6


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol

errors: list[str] = []
checks: list[str] = []

lineage = read_csv("authoritative_metric_lineage.csv")
if len(lineage) != 18:
    errors.append(f"Expected 18 authoritative model rows; found {len(lineage)}")
else:
    checks.append("Authoritative metric lineage contains 18 model rows")

by_ds: dict[str, list[dict[str, str]]] = defaultdict(list)
core_lookup: dict[tuple[str, str], float] = {}
for row in lineage:
    ds, model = row["dataset"], row["model"]
    by_ds[ds].append(row)
    test_f1 = float(row["test_f1"])
    pr_auc = float(row["test_pr_auc"])
    val_f1 = float(row["validation_f1"])
    fpr, fnr = float(row["fpr"]), float(row["fnr"])
    q = math.sqrt(test_f1 * pr_auc)
    o = 1.0 - (0.70 * fnr + 0.30 * fpr)
    g = 1.0 - max(0.0, val_f1 - test_f1)
    core = (q ** 0.45) * (o ** 0.35) * (g ** 0.20)
    core_lookup[(ds, model)] = core
    for field, value in (("Q", q), ("O", o), ("G", g), ("core", core)):
        if not close(float(row[field]), value):
            errors.append(f"{ds}/{model}: {field} mismatch: file={row[field]} computed={value:.9f}")
if not errors:
    checks.append("Q, O, G, and Core reproduce from the frozen metric matrix")

expected_val = {"CICIDS2017": "LightGBM", "NSL-KDD": "LightGBM", "UNSW-NB15": "Random Forest"}
expected_test = {"CICIDS2017": "LightGBM", "NSL-KDD": "XGBoost", "UNSW-NB15": "XGBoost"}
for ds, rows in by_ds.items():
    val_winner = max(rows, key=lambda r: float(r["validation_f1"]))["model"]
    test_winner = max(rows, key=lambda r: float(r["test_f1"]))["model"]
    if val_winner != expected_val[ds]:
        errors.append(f"{ds}: validation winner {val_winner}, expected {expected_val[ds]}")
    if test_winner != expected_test[ds]:
        errors.append(f"{ds}: test winner {test_winner}, expected {expected_test[ds]}")
checks.append("Validation and held-out winners match the frozen UNSW/NSL/CIC lineage")

stage = read_csv("frozen_stage2_scores.csv")
by_stage: dict[str, list[dict[str, str]]] = defaultdict(list)
for row in stage:
    ds, model = row["dataset"], row["model"]
    by_stage[ds].append(row)
    core = core_lookup[(ds, model)]
    e, t = float(row["E"]), float(row["T"])
    score = (core ** 0.60) * (e ** 0.25) * (t ** 0.15)
    if not close(float(row["core"]), core):
        errors.append(f"{ds}/{model}: Stage-2 Core mismatch")
    if not close(float(row["stage2"]), score):
        errors.append(f"{ds}/{model}: Stage-2 mismatch: file={row['stage2']} computed={score:.9f}")
for ds, rows in by_stage.items():
    ordered = sorted(rows, key=lambda r: float(r["stage2"]), reverse=True)
    for rank, row in enumerate(ordered, 1):
        if int(row["rank"]) != rank:
            errors.append(f"{ds}/{row['model']}: rank {row['rank']}, computed {rank}")
checks.append("Stage-2 scores and ranks reproduce from frozen Core, E, and T")

meta = json.loads((RESULTS / "frozen_release_metadata.json").read_text(encoding="utf-8"))
for ds, rows in by_stage.items():
    winner = max(rows, key=lambda r: float(r["stage2"]))["model"]
    if meta["stage2_winners"].get(ds) != winner:
        errors.append(f"{ds}: metadata Stage-2 winner mismatch")
checks.append("Release metadata agrees with score tables")

manifest = RESULTS / "frozen_generated_manifest.sha256"
for i, line in enumerate(manifest.read_text(encoding="utf-8").splitlines()):
    if i == 0 or not line.strip():
        continue
    digest, rel = line.split("  ", 1)
    path = ROOT / rel
    if not path.exists():
        errors.append(f"Manifest file missing: {rel}")
        continue
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != digest:
        errors.append(f"Manifest digest mismatch: {rel}")
checks.append("All generated-file SHA-256 values match the manifest")

report = {
    "status": "PASS" if not errors else "FAIL",
    "checks": checks,
    "errors": errors,
}
(RESULTS / "frozen_validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
raise SystemExit(0 if not errors else 1)
