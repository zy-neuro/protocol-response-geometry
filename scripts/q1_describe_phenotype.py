#!/usr/bin/env python3
"""Q1 DESCRIBE: cellular phenotype organization across regimes.

No molecular or Cre labels enter the statistics.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PHENO = ROOT / "results/phenotype"
OUT = ROOT / "results/q1"
SEED = 0
N_SHUFFLE = 100
KNN = 15
VARIANCE_FLOOR = 0.80
FIXED_K = 5
N_PAIR_SAMPLE = 80000

SCORE_FILES = {
    "long_pulse": PHENO / "long_pulse_pca_scores.csv",
    "ramp": PHENO / "ramp_pca_scores.csv",
    "brief_pulse": PHENO / "brief_pulse_pca_scores.csv",
}

VAR_FILES = {
    "long_pulse": PHENO / "long_pulse_pca_variance.csv",
    "ramp": PHENO / "ramp_pca_variance.csv",
    "brief_pulse": PHENO / "brief_pulse_pca_variance.csv",
}

DISPLAY = {
    "long_pulse": "Long-pulse",
    "ramp": "Ramp",
    "brief_pulse": "Brief-pulse",
}


def pc_cols(fieldnames: list[str]) -> list[str]:
    cols = [c for c in fieldnames if c.startswith("PC") and c[2:].isdigit()]
    return sorted(cols, key=lambda c: int(c[2:]))


def parse_session(row: dict) -> str:
    ses = row.get("session_id") or ""
    if ses:
        return ses
    fs = row.get("file_stem") or ""
    if "_ses-" in fs:
        return fs.split("_ses-")[1].replace("_icephys", "")
    return ""


def load_variance(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def choose_k(var_rows: list[dict], floor: float, n_available: int) -> int:
    k = n_available
    for row in var_rows:
        try:
            cum = float(row["cumulative_explained_variance_ratio"])
            comp = int(float(row["component"]))
        except (KeyError, ValueError):
            continue
        if cum >= floor:
            k = comp
            break
    return max(1, min(k, n_available))


def load_scores(path: Path) -> tuple[list[str], np.ndarray]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = pc_cols(reader.fieldnames or [])
        stems: list[str] = []
        mats: list[list[float]] = []
        for row in reader:
            fs = row.get("file_stem") or ""
            if not fs:
                ses = parse_session(row)
                fs = f"ses-{ses}" if ses else ""
            if not fs:
                continue
            stems.append(fs)
            mats.append([float(row[c]) for c in cols])
    return stems, np.asarray(mats, dtype=float)


def pairwise_euclidean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError(f"bad phenotype matrix shape {x.shape}")
    nrm = np.sum(x * x, axis=1, keepdims=True)
    d2 = nrm + nrm.T - 2.0 * np.dot(x, x.T)
    np.maximum(d2, 0.0, out=d2)
    d = np.sqrt(d2)
    np.fill_diagonal(d, 0.0)
    return d


def sample_pairs(n: int, rng: np.random.Generator, n_pairs: int = N_PAIR_SAMPLE) -> tuple[np.ndarray, np.ndarray]:
    max_pairs = n * (n - 1) // 2
    take = min(n_pairs, max_pairs)
    # sample unique unordered pairs via random i<j
    i = rng.integers(0, n, size=take * 3)
    j = rng.integers(0, n, size=take * 3)
    mask = i < j
    i, j = i[mask], j[mask]
    if i.size < take:
        ii, jj = np.triu_indices(n, k=1)
        pick = rng.choice(ii.size, size=take, replace=False)
        return ii[pick], jj[pick]
    return i[:take], j[:take]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if ra.size < 3:
        return float("nan")
    return float(np.corrcoef(ra.astype(float), rb.astype(float))[0, 1])


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def knn_indices(d: np.ndarray, k: int) -> np.ndarray:
    n = d.shape[0]
    k = min(k, n - 1)
    out = np.empty((n, k), dtype=int)
    for i in range(n):
        order = np.argpartition(d[i], k + 1)
        neigh = [j for j in order[: k + 1] if j != i]
        if len(neigh) < k:
            full = np.argsort(d[i])
            neigh = [j for j in full if j != i][:k]
        out[i] = np.array(neigh[:k], dtype=int)
    return out


def knn_jaccard(idx_a: np.ndarray, idx_b: np.ndarray) -> np.ndarray:
    n = idx_a.shape[0]
    jac = np.empty(n, dtype=float)
    for i in range(n):
        sa = set(idx_a[i].tolist())
        sb = set(idx_b[i].tolist())
        union = sa | sb
        jac[i] = (len(sa & sb) / len(union)) if union else 0.0
    return jac


def align_pair(
    stems_a: list[str],
    xa: np.ndarray,
    stems_b: list[str],
    xb: np.ndarray,
    k_a: int,
    k_b: int,
    rng: np.random.Generator,
) -> dict:
    pos_a = {s: i for i, s in enumerate(stems_a)}
    pos_b = {s: i for i, s in enumerate(stems_b)}
    shared = [s for s in stems_b if s in pos_a]
    ia = np.array([pos_a[s] for s in shared], dtype=int)
    ib = np.array([pos_b[s] for s in shared], dtype=int)

    da = pairwise_euclidean(xa[ia, :k_a])
    db = pairwise_euclidean(xb[ib, :k_b])
    n = da.shape[0]
    ii, jj = sample_pairs(n, rng)
    ua, ub = da[ii, jj], db[ii, jj]
    rho = spearman(ua, ub)
    r = pearson(ua, ub)

    null = np.empty(N_SHUFFLE, dtype=float)
    for t in range(N_SHUFFLE):
        perm = rng.permutation(n)
        null[t] = spearman(ua, db[perm[ii], perm[jj]])
    p = (1.0 + np.sum(null >= rho)) / (1.0 + N_SHUFFLE)

    knn_a = knn_indices(da, KNN)
    knn_b = knn_indices(db, KNN)
    jac = knn_jaccard(knn_a, knn_b)

    # sign-invariant PC1 correspondence
    s1 = xa[ia, 0]
    t1 = xb[ib, 0]
    pc1_abs_r = abs(pearson(s1, t1))

    return {
        "n_shared": int(len(shared)),
        "k_a": int(k_a),
        "k_b": int(k_b),
        "spearman_D": rho,
        "pearson_D": r,
        "null_spearman_mean": float(np.mean(null)),
        "null_spearman_sd": float(np.std(null)),
        "empirical_p": float(p),
        "knn_jaccard_mean": float(np.mean(jac)),
        "knn_jaccard_median": float(np.median(jac)),
        "pc1_abs_pearson": float(pc1_abs_r),
        "mean_distance_a": float(np.mean(ua)),
        "mean_distance_b": float(np.mean(ub)),
    }


def within_protocol_stats(x: np.ndarray, k: int) -> dict:
    xk = x[:, :k]
    d = pairwise_euclidean(xk)
    n = xk.shape[0]
    ii, jj = np.triu_indices(n, k=1)
    u = d[ii, jj]
    knn = knn_indices(d, KNN)
    # local compactness: mean distance to kNN / mean global pairwise
    local = np.empty(n, dtype=float)
    for i in range(n):
        local[i] = float(np.mean(d[i, knn[i]]))
    return {
        "n_cells": int(n),
        "k": int(k),
        "mean_pairwise_distance": float(np.mean(u)),
        "median_pairwise_distance": float(np.median(u)),
        "mean_knn_distance": float(np.mean(local)),
        "knn_over_global": float(np.mean(local) / np.mean(u)) if np.mean(u) else float("nan"),
        "score_sd": [float(np.std(xk[:, j])) for j in range(k)],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    protocols = list(SCORE_FILES)
    stems: dict[str, list[str]] = {}
    X: dict[str, np.ndarray] = {}
    k_var: dict[str, int] = {}
    variance: dict[str, list[dict]] = {}

    for p in protocols:
        s, mat = load_scores(SCORE_FILES[p])
        stems[p] = s
        X[p] = mat
        variance[p] = load_variance(VAR_FILES[p])
        k_var[p] = choose_k(variance[p], VARIANCE_FLOOR, mat.shape[1])

    # within-protocol
    within_rows = []
    for p in protocols:
        st_var = within_protocol_stats(X[p], k_var[p])
        st_fix = within_protocol_stats(X[p], min(FIXED_K, X[p].shape[1]))
        ve = []
        for row in variance[p]:
            ve.append(
                {
                    "protocol": p,
                    "PC": row.get("PC", ""),
                    "component": row.get("component", ""),
                    "explained_variance_ratio": row.get("explained_variance_ratio", ""),
                    "cumulative_explained_variance_ratio": row.get(
                        "cumulative_explained_variance_ratio", ""
                    ),
                }
            )
        within_rows.append(
            {
                "protocol": p,
                "n_cells": st_var["n_cells"],
                "k_variance_80": st_var["k"],
                "k_fixed": st_fix["k"],
                "mean_pairwise_k80": st_var["mean_pairwise_distance"],
                "mean_knn_k80": st_var["mean_knn_distance"],
                "knn_over_global_k80": st_var["knn_over_global"],
                "mean_pairwise_k5": st_fix["mean_pairwise_distance"],
                "knn_over_global_k5": st_fix["knn_over_global"],
            }
        )
        with open(OUT / f"q1_variance_{p}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ve[0].keys()))
            w.writeheader()
            w.writerows(ve)

    with open(OUT / "q1_within_protocol.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(within_rows[0].keys()))
        w.writeheader()
        w.writerows(within_rows)

    pairs = [
        ("long_pulse", "ramp"),
        ("long_pulse", "brief_pulse"),
        ("ramp", "brief_pulse"),
    ]
    pair_rows = []
    for a, b in pairs:
        for tag, ka, kb in (
            ("k80", k_var[a], k_var[b]),
            ("k5", min(FIXED_K, X[a].shape[1]), min(FIXED_K, X[b].shape[1])),
        ):
            stats = align_pair(stems[a], X[a], stems[b], X[b], ka, kb, rng)
            stats.update({"pair": f"{a}__{b}", "k_rule": tag})
            pair_rows.append(stats)

    with open(OUT / "q1_cross_protocol.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        w.writerows(pair_rows)

    # triple overlap (Route I Φ)
    set_map = {p: set(stems[p]) for p in protocols}
    triple = sorted(set_map["long_pulse"] & set_map["ramp"] & set_map["brief_pulse"])
    triple_path = OUT / "q1_triple_cells.txt"
    triple_path.write_text("\n".join(triple) + "\n")

    summary = {
        "question": "Q1",
        "labels_used": "none",
        "variance_floor": VARIANCE_FLOOR,
        "fixed_k": FIXED_K,
        "knn": KNN,
        "n_shuffle": N_SHUFFLE,
        "seed": SEED,
        "k_variance_80": k_var,
        "n_cells": {p: len(stems[p]) for p in protocols},
        "n_overlap": {
            "long_pulse__ramp": len(set_map["long_pulse"] & set_map["ramp"]),
            "long_pulse__brief_pulse": len(set_map["long_pulse"] & set_map["brief_pulse"]),
            "ramp__brief_pulse": len(set_map["ramp"] & set_map["brief_pulse"]),
            "all_three": len(triple),
        },
        "route": "I_for_LP_Ramp_and_triple_subset",
        "within_protocol": within_rows,
        "cross_protocol": pair_rows,
    }
    (OUT / "q1_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # figures
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    xs = np.arange(1, 11)
    for p in protocols:
        ratios = []
        for row in variance[p]:
            ratios.append(float(row["explained_variance_ratio"]))
        ax.plot(
            range(1, len(ratios) + 1),
            np.cumsum(ratios),
            marker="o",
            linewidth=1.5,
            label=DISPLAY[p],
        )
    ax.axhline(VARIANCE_FLOOR, color="0.5", linestyle="--", linewidth=1)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Cumulative variance explained")
    ax.set_title("Q1: Dimensional organization by protocol")
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(xs)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "q1_variance_cumulative.pdf")
    fig.savefig(OUT / "q1_variance_cumulative.png", dpi=160)
    plt.close(fig)

    # bar of D-alignment (k80)
    k80 = [r for r in pair_rows if r["k_rule"] == "k80"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    labels = [r["pair"].replace("__", " vs\n").replace("_", "-") for r in k80]
    vals = [r["spearman_D"] for r in k80]
    nulls = [r["null_spearman_mean"] for r in k80]
    x = np.arange(len(k80))
    ax.bar(x - 0.18, vals, width=0.36, label="Observed Spearman(D,D)")
    ax.bar(x + 0.18, nulls, width=0.36, label="Shuffle null mean")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[a] + "–" + DISPLAY[b] for a, b in pairs])
    ax.set_ylabel("Spearman correlation of pairwise distances")
    ax.set_title("Q1: Phenotypic conservation vs reconfiguration")
    ax.legend(frameon=False)
    ax.set_ylim(-0.15, 1.0)
    fig.tight_layout()
    fig.savefig(OUT / "q1_distance_alignment.pdf")
    fig.savefig(OUT / "q1_distance_alignment.png", dpi=160)
    plt.close(fig)

    # write readable landing
    lines = [
        "# Q1 results — phenotypic organization (no molecular labels)",
        "",
        "Object: protocol PC scores. Lens: Euclidean pairwise distances + kNN.",
        "",
        "## Within protocol",
    ]
    for row in within_rows:
        lines.append(
            f"- {DISPLAY[row['protocol']]}: n={row['n_cells']}, k80={row['k_variance_80']}, "
            f"knn/global={row['knn_over_global_k80']:.3f}"
        )
    lines += ["", "## Cross-protocol (shared cells, k80)", ""]
    for r, (a, b) in zip(k80, pairs):
        lines.append(
            f"- {DISPLAY[a]} vs {DISPLAY[b]}: n={r['n_shared']}, "
            f"Spearman(D)={r['spearman_D']:.3f} (null {r['null_spearman_mean']:.3f}, "
            f"p={r['empirical_p']:.4f}), kNN Jaccard={r['knn_jaccard_mean']:.3f}, "
            f"|r|(PC1)={r['pc1_abs_pearson']:.3f}"
        )
    lines += [
        "",
        f"Triple overlap n={len(triple)} (Route I subset).",
        "",
        "Landing terms: phenotypically conserved vs phenotypically reconfigured.",
        "Molecule/Cre not used.",
    ]
    (OUT / "q1_landing.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
