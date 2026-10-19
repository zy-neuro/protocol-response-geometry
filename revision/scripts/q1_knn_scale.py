#!/usr/bin/env python3
"""Cross-regime kNN Jaccard vs neighborhood size (revision analysis).

Reads submitted k80 phenotype scores. Writes chance-adjusted Jaccard
across k = 15, 30, 60, 120, 240, 480, 600, plus per-cell Jaccard at k=15.
Does not modify original results/q1.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ORIG_PHENO, OUT_Q1  # noqa: E402

SEED = 0
N_SHUFFLE = 100
K_GRID = (15, 30, 60, 120, 240, 480, 600)
VARIANCE_FLOOR = 0.80
PROTOCOLS = ("long_pulse", "ramp", "brief_pulse")
PAIRS = (
    ("long_pulse", "ramp"),
    ("long_pulse", "brief_pulse"),
    ("ramp", "brief_pulse"),
)


def pc_cols(fieldnames: list[str]) -> list[str]:
    cols = [c for c in fieldnames if c.startswith("PC") and c[2:].isdigit()]
    return sorted(cols, key=lambda c: int(c[2:]))


def load_scores(path: Path) -> tuple[list[str], np.ndarray]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = pc_cols(reader.fieldnames or [])
        stems, mats = [], []
        for row in reader:
            fs = row.get("file_stem") or ""
            if not fs:
                continue
            stems.append(fs)
            mats.append([float(row[c]) for c in cols])
    return stems, np.asarray(mats, dtype=float)


def load_k80(path: Path, n_available: int) -> int:
    k = n_available
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if float(row["cumulative_explained_variance_ratio"]) >= VARIANCE_FLOOR:
                k = int(float(row["component"]))
                break
    return max(1, min(k, n_available))


def pairwise_euclidean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    nrm = np.sum(x * x, axis=1, keepdims=True)
    d2 = nrm + nrm.T - 2.0 * np.dot(x, x.T)
    np.maximum(d2, 0.0, out=d2)
    d = np.sqrt(d2)
    np.fill_diagonal(d, 0.0)
    return d


def knn_indices(d: np.ndarray, k: int) -> np.ndarray:
    n = d.shape[0]
    k = min(k, n - 1)
    idx = np.argpartition(d, kth=k, axis=1)[:, : k + 1]
    out = np.empty((n, k), dtype=np.int32)
    for i in range(n):
        row = idx[i]
        row = row[row != i]
        if row.size < k:
            full = np.argsort(d[i])
            row = full[full != i][:k]
        out[i] = row[:k]
    return out


def jaccard_rows(idx_a: np.ndarray, idx_b: np.ndarray) -> np.ndarray:
    n, k = idx_a.shape
    jac = np.empty(n, dtype=float)
    for i in range(n):
        inter = np.intersect1d(idx_a[i], idx_b[i], assume_unique=False).size
        union = 2 * k - inter
        jac[i] = (inter / union) if union else 0.0
    return jac


def align_shared(stems_a, xa, stems_b, xb, ka, kb):
    pos_a = {s: i for i, s in enumerate(stems_a)}
    shared = [s for s in stems_b if s in pos_a]
    ia = np.array([pos_a[s] for s in shared], dtype=int)
    pos_b = {s: i for i, s in enumerate(stems_b)}
    ib = np.array([pos_b[s] for s in shared], dtype=int)
    da = pairwise_euclidean(xa[ia, :ka])
    db = pairwise_euclidean(xb[ib, :kb])
    return shared, da, db


def main() -> None:
    OUT_Q1.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    stems, X, k80 = {}, {}, {}
    for p in PROTOCOLS:
        s, mat = load_scores(ORIG_PHENO / f"{p}_pca_scores.csv")
        stems[p], X[p] = s, mat
        k80[p] = load_k80(ORIG_PHENO / f"{p}_pca_variance.csv", mat.shape[1])
        print(f"{p}: n={len(s)} k80={k80[p]}", flush=True)

    scale_rows = []
    cell_rows = []
    for a, b in PAIRS:
        pair = f"{a}__{b}"
        shared, da, db = align_shared(stems[a], X[a], stems[b], X[b], k80[a], k80[b])
        n = da.shape[0]
        print(f"{pair}: n_shared={n}", flush=True)
        for k in K_GRID:
            kk = min(k, n - 1)
            knn_a = knn_indices(da, kk)
            knn_b = knn_indices(db, kk)
            jac = jaccard_rows(knn_a, knn_b)
            obs_mean = float(np.mean(jac))
            obs_median = float(np.median(jac))
            null_means = np.empty(N_SHUFFLE, dtype=float)
            for t in range(N_SHUFFLE):
                perm = rng.permutation(n)
                null_means[t] = float(np.mean(jaccard_rows(knn_a, knn_b[perm])))
            chance_mean = float(np.mean(null_means))
            chance_sd = float(np.std(null_means))
            scale_rows.append(
                {
                    "pair": pair,
                    "k": int(kk),
                    "n_shared": int(n),
                    "observed_jaccard_mean": obs_mean,
                    "observed_jaccard_median": obs_median,
                    "chance_jaccard_mean": chance_mean,
                    "chance_jaccard_sd": chance_sd,
                    "adjusted_jaccard": obs_mean - chance_mean,
                    "n_shuffle": N_SHUFFLE,
                }
            )
            print(
                f"  k={kk:3d} obs={obs_mean:.4f} chance={chance_mean:.4f} "
                f"adj={obs_mean - chance_mean:.4f}",
                flush=True,
            )
            if kk == 15:
                for stem, val in zip(shared, jac):
                    cell_rows.append({"pair": pair, "file_stem": stem, "jaccard_k15": float(val)})

    fields = list(scale_rows[0].keys())
    with open(OUT_Q1 / "q1_knn_scale.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(scale_rows)

    with open(OUT_Q1 / "q1_knn_jaccard_k15.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "file_stem", "jaccard_k15"])
        w.writeheader()
        w.writerows(cell_rows)

    print(f"Wrote {OUT_Q1}")


if __name__ == "__main__":
    main()
