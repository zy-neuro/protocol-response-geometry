#!/usr/bin/env python3
"""Dump protocol PCA loadings from submitted IPFX features.

Same filters as scripts/pipeline/02_build_phenotype.py. Writes loadings only;
does not overwrite original score/variance tables.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ORIG_FEATURES, ORIG_PHENO, OUT_PHENO  # noqa: E402

MAX_MISSING = 0.30
PROTOCOL_PREFIX = {
    "long_pulse": "feat.long_squares.",
    "ramp": "feat.ramps.",
    "brief_pulse": "feat.short_squares.",
}


def load_features(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("extract_ok") == "1"]
        fields = list(rows[0].keys()) if rows else []
    return fields, rows


def numeric_matrix(rows: list[dict], cols: list[str]) -> np.ndarray:
    mat = np.full((len(rows), len(cols)), np.nan, dtype=float)
    for i, row in enumerate(rows):
        for j, c in enumerate(cols):
            v = row.get(c, "")
            if v in ("", None):
                continue
            try:
                mat[i, j] = float(v)
            except ValueError:
                continue
    return mat


def pca_loadings(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0)
    Xf = np.where(np.isfinite(X), X, mu)
    Xc = Xf - Xf.mean(axis=0, keepdims=True)
    std = Xc.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    Z = Xc / std
    u, s, vt = np.linalg.svd(Z, full_matrices=False)
    scores = u * s
    var = (s**2) / max(1, Z.shape[0] - 1)
    var = var / var.sum()
    loadings = vt.T  # features x PCs
    return scores, var, loadings


def pretty_feature(col: str, prefix: str) -> str:
    name = col[len(prefix) :] if col.startswith(prefix) else col
    return name.replace("_", " ")


def main() -> None:
    src = ORIG_FEATURES / "ipfx_cell_features.csv"
    if not src.exists():
        raise SystemExit(f"Missing {src}")
    fields, rows = load_features(src)
    OUT_PHENO.mkdir(parents=True, exist_ok=True)
    summary = json.loads((ORIG_PHENO / "phenotype_summary.json").read_text())
    meta = {}
    for proto, prefix in PROTOCOL_PREFIX.items():
        expected = list(summary[proto]["features"])
        cols = [c for c in fields if c.startswith(prefix)]
        X = numeric_matrix(rows, cols)
        keep = []
        for j, c in enumerate(cols):
            miss = float(np.mean(~np.isfinite(X[:, j])))
            if miss <= MAX_MISSING and np.nanstd(X[:, j]) > 0:
                keep.append(j)
        cols = [cols[j] for j in keep]
        X = X[:, keep]
        present = np.isfinite(X).any(axis=1)
        X = X[present]
        if cols != expected:
            print(f"WARNING {proto}: kept {len(cols)} features vs summary {len(expected)}")
        _, var, loadings = pca_loadings(X)
        n_pc = min(loadings.shape[1], 5)
        out = OUT_PHENO / f"{proto}_pca_loadings.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "feature",
                    "display",
                    *[f"PC{i}" for i in range(1, n_pc + 1)],
                    "abs_PC1",
                ]
            )
            order = np.argsort(-np.abs(loadings[:, 0]))
            for j in order:
                w.writerow(
                    [
                        cols[j],
                        pretty_feature(cols[j], prefix),
                        *[float(loadings[j, i]) for i in range(n_pc)],
                        float(abs(loadings[j, 0])),
                    ]
                )
        meta[proto] = {
            "n_cells": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "pc1_var": float(var[0]),
            "top10_pc1": [
                pretty_feature(cols[j], prefix) for j in order[:10]
            ],
            "path": str(out),
        }
        print(f"{proto}: n={X.shape[0]} p={X.shape[1]} PC1 var={var[0]:.3f} -> {out}", flush=True)
    (OUT_PHENO / "loadings_summary.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
