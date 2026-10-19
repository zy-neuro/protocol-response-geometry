#!/usr/bin/env python3
"""Q3b/c EXPLAIN: transcriptomic dimensions and pre-specified modules.

Q3b: RNA PCA → type-centroid distances T(a,b) vs Q3a S_p / pair tags.
Q3c: few pre-specified excitability / marker programs only — no genome-wide fishing.

Uses Q3a pair tags in results/q3a/.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-prg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JOIN = ROOT / "results/m0/m0_molecular_join.csv"
PAIRS = ROOT / "results/q3a/q3a_pair_tags.csv"
GENE_LIST = ROOT / "configs/q3_curated_excitability_genes.txt"
OUT_B = ROOT / "results/q3b"
OUT_C = ROOT / "results/q3c"
NEMO_COUNTS_NAME = "20200513_Mouse_PatchSeq_Release_count.v2.csv"


def nemo_counts_csv() -> Path:
    root = ROOT / "data/official/nemo"
    hits = sorted(p for p in root.rglob(NEMO_COUNTS_NAME) if p.is_file()) if root.exists() else []
    if not hits:
        raise SystemExit(
            f"Missing {NEMO_COUNTS_NAME} under {root}. "
            "Place the official NeMO Patch-seq count matrix there to re-run Q3b/c. "
            "Figure panels use the bundled tables in results/q3b/ and results/q3c/."
        )
    return hits[0]

SEED = 0
N_SHUFFLE = 50
N_PCS = 10
N_HVG = 500
N_PRIMARY = 500

# Pre-specified programs (Q3c). Marker modules are controls, not discoveries.
MODULES = {
    "Nav": ["Scn1a", "Scn2a1", "Scn3a", "Scn8a", "Scn9a", "Scn1b", "Scn2b", "Scn3b", "Scn4b"],
    "Kv3": ["Kcnc1", "Kcnc2", "Kcnc3", "Kcnc4"],
    "Kv_delayed": ["Kcna1", "Kcna2", "Kcnb1", "Kcnb2", "Kcnq2", "Kcnq3", "Kcnq5"],
    "HCN": ["Hcn1", "Hcn2", "Hcn3", "Hcn4"],
    "Cav": ["Cacna1a", "Cacna1b", "Cacna1c", "Cacna1d", "Cacna1e", "Cacna1g", "Cacna1h", "Cacna1i"],
    "SK_BK": ["Kcnn1", "Kcnn2", "Kcnn3", "Kcnma1", "Kcnmb1", "Kcnmb2", "Kcnmb4"],
    "K_leak": ["Kcnk1", "Kcnk2", "Kcnk3", "Kcnk9", "Kcnj2", "Kcnj4", "Kcnj12"],
    "GABA_A": ["Gabra1", "Gabra2", "Gabra3", "Gabra4", "Gabra5", "Gabrb1", "Gabrb2", "Gabrb3", "Gabrg1", "Gabrg2", "Gabrg3"],
    "AMPA_NMDA": ["Gria1", "Gria2", "Gria3", "Gria4", "Grin1", "Grin2a", "Grin2b", "Grin2c", "Grin2d"],
    "Pvalb_marker": ["Pvalb", "Tac1"],
    "Sst_marker": ["Sst", "Crh", "Cort", "Chodl"],
    "Vip_marker": ["Vip", "Cck", "Calb2"],
}
MARKER_MODULES = {"Pvalb_marker", "Sst_marker", "Vip_marker"}


def read_gene_list(path: Path) -> list[str]:
    genes = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            genes.append(s)
    return genes


def load_join(path: Path) -> dict[str, str]:
    sample_to_type = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            sid = (row.get("transcriptomics_sample_id") or "").strip()
            tt = (row.get("t_type") or "").strip()
            if sid and tt:
                sample_to_type[sid] = tt
    return sample_to_type


def load_pairs(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in (
            "S_long_pulse",
            "S_ramp",
            "S_brief_pulse",
            "rank_norm_long_pulse",
            "rank_norm_ramp",
            "rank_norm_brief_pulse",
            "rank_mean",
            "rank_cv",
            "rank_range",
        ):
            r[k] = float(r[k])
    return rows


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def spearman_p(rho: float, n: int) -> float:
    if not np.isfinite(rho) or n < 3 or abs(rho) >= 1:
        return 0.0 if np.isfinite(rho) and abs(rho) >= 1 else float("nan")
    t = abs(rho) * math.sqrt((n - 2) / (1.0 - rho * rho))
    # normal tail; df is large enough for ranking
    return float(math.erfc(t / math.sqrt(2.0)))


def bh_fdr(pvals: list[float]) -> list[float]:
    n = len(pvals)
    order = np.argsort(pvals)
    fdr = np.empty(n, dtype=float)
    prev = 1.0
    for rank_from_end, idx in enumerate(order[::-1]):
        rank = n - rank_from_end
        val = pvals[idx] * n / rank
        prev = min(prev, val)
        fdr[idx] = min(1.0, prev)
    return fdr.tolist()


def mannwhitney_p(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_perm: int = 200) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 3 or b.size < 3:
        return float("nan")
    obs = float(np.mean(a) - np.mean(b))
    pool = np.concatenate([a, b])
    n_a = a.size
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        delta = float(pool[:n_a].mean() - pool[n_a:].mean())
        if abs(delta) >= abs(obs):
            count += 1
    return (1.0 + count) / (1.0 + n_perm)


def pairwise_euclid(M: np.ndarray) -> np.ndarray:
    nrm = np.sum(M * M, axis=1, keepdims=True)
    d2 = nrm + nrm.T - 2.0 * (M @ M.T)
    np.maximum(d2, 0.0, out=d2)
    D = np.sqrt(d2)
    np.fill_diagonal(D, 0.0)
    return D


def pca_scores(X: np.ndarray, n_pcs: int) -> tuple[np.ndarray, np.ndarray]:
    Xc = X - X.mean(axis=0, keepdims=True)
    std = Xc.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    Z = Xc / std
    # economy SVD: cells x genes
    u, s, vt = np.linalg.svd(Z, full_matrices=False)
    k = min(n_pcs, s.size)
    scores = u[:, :k] * s[:k]
    var = (s**2) / max(1, Z.shape[0] - 1)
    var = var / var.sum()
    return scores[:, :k], var[:k]


def load_counts(path: Path, keep_samples: list[str]) -> tuple[list[str], np.ndarray, list[str]]:
    keep_set = set(keep_samples)
    with open(path, newline="") as f:
        header = next(csv.reader(f))
    all_samples = [s.strip().strip('"') for s in header[1:]]
    col_idx = [i for i, s in enumerate(all_samples) if s in keep_set]
    ordered = [all_samples[i] for i in col_idx]
    genes: list[str] = []
    blocks: list[np.ndarray] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        chunk = []
        chunk_genes = []
        for row in reader:
            gene = row[0].strip().strip('"')
            vals = np.fromiter((float(row[i + 1]) for i in col_idx), dtype=np.float32, count=len(col_idx))
            chunk_genes.append(gene)
            chunk.append(vals)
            if len(chunk) >= 2000:
                genes.extend(chunk_genes)
                blocks.append(np.vstack(chunk))
                chunk, chunk_genes = [], []
        if chunk:
            genes.extend(chunk_genes)
            blocks.append(np.vstack(chunk))
    mat = np.vstack(blocks) if blocks else np.zeros((0, len(ordered)), dtype=np.float32)
    return genes, mat, ordered


def log_cpm(counts: np.ndarray) -> np.ndarray:
    lib = counts.sum(axis=0)
    lib[lib == 0] = 1.0
    return np.log1p(np.float32(1.0e6) * counts / lib)


def type_means(X: np.ndarray, labels: list[str], types: list[str]) -> np.ndarray:
    mus = []
    for t in types:
        idx = np.array([i for i, lab in enumerate(labels) if lab == t], dtype=int)
        mus.append(X[idx].mean(axis=0) if idx.size else np.full(X.shape[1], np.nan))
    return np.asarray(mus, dtype=float)


def pair_T(types: list[str], D: np.ndarray, pairs: list[dict]) -> np.ndarray:
    index = {t: i for i, t in enumerate(types)}
    out = np.full(len(pairs), np.nan)
    for k, r in enumerate(pairs):
        i, j = index.get(r["type_a"]), index.get(r["type_b"])
        if i is not None and j is not None:
            out[k] = D[i, j]
    return out


def type_selectivity(types: list[str], pairs: list[dict]) -> dict[str, dict]:
    stats = {t: {"sel": 0, "stab": 0, "n": 0, "rank_range": []} for t in types}
    for r in pairs:
        for key in ("type_a", "type_b"):
            t = r[key]
            if t not in stats:
                continue
            stats[t]["n"] += 1
            if r["tag"] == "selective":
                stats[t]["sel"] += 1
            elif r["tag"] == "stable":
                stats[t]["stab"] += 1
            stats[t]["rank_range"].append(r["rank_range"])
    out = {}
    for t, s in stats.items():
        n = max(1, s["n"])
        out[t] = {
            "frac_selective_pairs": s["sel"] / n,
            "frac_stable_pairs": s["stab"] / n,
            "selectivity_index": (s["sel"] - s["stab"]) / n,
            "mean_pair_rank_range": float(np.mean(s["rank_range"])) if s["rank_range"] else float("nan"),
        }
    return out


def empirical_spearman_p(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n: int) -> float:
    obs = spearman(x, y)
    if not np.isfinite(obs):
        return float("nan")
    count = 0
    y_work = y.copy()
    for _ in range(n):
        rng.shuffle(y_work)
        if abs(spearman(x, y_work)) >= abs(obs):
            count += 1
    return (1.0 + count) / (1.0 + n)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run_panel(
    name: str,
    expr: np.ndarray,
    samples: list[str],
    labels: list[str],
    types: list[str],
    pairs: list[dict],
    sel: dict[str, dict],
    genes: list[str],
    rng: np.random.Generator,
    out: Path,
    gene_table: bool,
) -> dict:
    scores, var = pca_scores(expr.T, N_PCS)  # cells x genes
    mus = type_means(scores, labels, types)
    T = pairwise_euclid(mus)
    t_pairs = pair_T(types, T, pairs)
    mean_s = np.array(
        [(r["S_long_pulse"] + r["S_ramp"] + r["S_brief_pulse"]) / 3.0 for r in pairs], dtype=float
    )
    rank_range = np.array([r["rank_range"] for r in pairs], dtype=float)
    tags = [r["tag"] for r in pairs]
    stab = t_pairs[np.array(tags) == "stable"]
    sele = t_pairs[np.array(tags) == "selective"]

    rho_s = spearman(t_pairs, mean_s)
    rho_r = spearman(t_pairs, rank_range)
    p_s = spearman_p(rho_s, int(np.isfinite(t_pairs).sum()))
    p_r = spearman_p(rho_r, int(np.isfinite(t_pairs).sum()))
    emp_s = empirical_spearman_p(t_pairs, mean_s, rng, N_SHUFFLE)
    emp_r = empirical_spearman_p(t_pairs, rank_range, rng, N_SHUFFLE)
    mw_p = mannwhitney_p(sele, stab, rng)

    pair_rows = []
    for r, tv in zip(pairs, t_pairs):
        row = dict(r)
        row[f"T_{name}"] = float(tv)
        pair_rows.append(row)
    write_csv(out / f"q3b_pairs_with_T_{name}.csv", pair_rows)

    t_rows = []
    for i, a in enumerate(types):
        for j, b in enumerate(types):
            if i < j:
                t_rows.append({"t_type_a": a, "t_type_b": b, "T": float(T[i, j])})
    write_csv(out / f"q3b_T_{name}.csv", t_rows)

    type_rows = []
    rna_pc_corr = []
    sel_idx = np.array([sel[t]["selectivity_index"] for t in types], dtype=float)
    for pc in range(mus.shape[1]):
        rho = spearman(mus[:, pc], sel_idx)
        rna_pc_corr.append({"RNA_PC": pc + 1, "spearman": rho, "p": spearman_p(rho, len(types))})
    write_csv(out / f"q3b_rna_pc_vs_selectivity_{name}.csv", rna_pc_corr)

    for i, t in enumerate(types):
        rec = {"t_type": t, "subclass": t.split()[0], "n_rna_cells": int(labels.count(t))}
        rec.update(sel[t])
        for pc in range(mus.shape[1]):
            rec[f"RNA_PC{pc+1}"] = float(mus[i, pc])
        type_rows.append(rec)
    write_csv(out / f"q3b_type_selectivity_{name}.csv", type_rows)

    cell_rows = []
    for i, sid in enumerate(samples):
        rec = {f"RNA_PC{pc+1}": float(scores[i, pc]) for pc in range(scores.shape[1])}
        rec["t_type"] = labels[i]
        rec["sample_id"] = sid
        cell_rows.append(rec)
    write_csv(out / f"q3b_cell_rna_pcs_{name}.csv", cell_rows)

    if gene_table:
        type_expr = type_means(expr.T, labels, types)
        grow = []
        for g_i, g in enumerate(genes):
            rho = spearman(type_expr[:, g_i], sel_idx)
            grow.append({"gene": g, "spearman": rho, "p": spearman_p(rho, len(types))})
        grow = sorted(grow, key=lambda r: abs(r["spearman"]), reverse=True)
        fdrs = bh_fdr([r["p"] for r in grow])
        for r, f in zip(grow, fdrs):
            r["fdr"] = f
        write_csv(out / f"q3b_genes_vs_selectivity_{name}.csv", grow)

    summary = {
        "panel": name,
        "n_genes": int(expr.shape[0]),
        "n_cells": int(expr.shape[1]),
        "n_types": len(types),
        "n_pairs": len(pairs),
        "var_explained_top5": [float(v) for v in var[:5]],
        "spearman_T_vs_meanS": rho_s,
        "p_T_vs_meanS": p_s,
        "empirical_p_T_vs_meanS": emp_s,
        "spearman_T_vs_rank_range": rho_r,
        "p_T_vs_rank_range": p_r,
        "empirical_p_T_vs_rank_range": emp_r,
        "mean_T_stable": float(np.nanmean(stab)),
        "mean_T_selective": float(np.nanmean(sele)),
        "mannwhitney_sel_vs_stab_p": mw_p,
    }
    return summary


def q3c_modules(
    expr_genes: list[str],
    expr: np.ndarray,
    labels: list[str],
    types: list[str],
    pairs: list[dict],
    sel: dict[str, dict],
    rng: np.random.Generator,
    out: Path,
) -> dict:
    gidx = {g: i for i, g in enumerate(expr_genes)}
    z = expr.copy()
    mu = z.mean(axis=1, keepdims=True)
    sd = z.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    z = (z - mu) / sd

    module_names = []
    type_scores = []
    used = {}
    for name, members in MODULES.items():
        present = [g for g in members if g in gidx]
        used[name] = present
        if len(present) < 2:
            continue
        idx = np.array([gidx[g] for g in present], dtype=int)
        cell_score = z[idx].mean(axis=0)
        mus = []
        for t in types:
            ii = np.array([i for i, lab in enumerate(labels) if lab == t], dtype=int)
            mus.append(float(cell_score[ii].mean()) if ii.size else float("nan"))
        module_names.append(name)
        type_scores.append(mus)
    M = np.asarray(type_scores, dtype=float).T  # types x modules
    sel_idx = np.array([sel[t]["selectivity_index"] for t in types], dtype=float)
    mean_s = np.array(
        [(r["S_long_pulse"] + r["S_ramp"] + r["S_brief_pulse"]) / 3.0 for r in pairs], dtype=float
    )
    rank_range = np.array([r["rank_range"] for r in pairs], dtype=float)
    tags = np.array([r["tag"] for r in pairs])

    type_rows = []
    for i, t in enumerate(types):
        rec = {"t_type": t, "subclass": t.split()[0], "selectivity_index": float(sel_idx[i])}
        for j, name in enumerate(module_names):
            rec[name] = float(M[i, j])
        type_rows.append(rec)
    write_csv(out / "q3c_type_module_scores.csv", type_rows)

    score_rows = []
    pair_rows = []
    for j, name in enumerate(module_names):
        rho_type = spearman(M[:, j], sel_idx)
        D = np.abs(M[:, j][:, None] - M[:, j][None, :])
        t_pairs = pair_T(types, D, pairs)
        rho_s = spearman(t_pairs, mean_s)
        rho_r = spearman(t_pairs, rank_range)
        stab = t_pairs[tags == "stable"]
        sele = t_pairs[tags == "selective"]
        score_rows.append(
            {
                "module": name,
                "role": "marker_control" if name in MARKER_MODULES else "excitability",
                "n_genes_present": len(used[name]),
                "genes": ";".join(used[name]),
                "spearman_score_vs_selectivity": rho_type,
                "p_score_vs_selectivity": spearman_p(rho_type, len(types)),
                "spearman_D_vs_meanS": rho_s,
                "p_D_vs_meanS": spearman_p(rho_s, len(pairs)),
                "empirical_p_D_vs_meanS": empirical_spearman_p(t_pairs, mean_s, rng, N_SHUFFLE),
                "spearman_D_vs_rank_range": rho_r,
                "p_D_vs_rank_range": spearman_p(rho_r, len(pairs)),
                "empirical_p_D_vs_rank_range": empirical_spearman_p(t_pairs, rank_range, rng, N_SHUFFLE),
                "mean_D_stable": float(np.nanmean(stab)),
                "mean_D_selective": float(np.nanmean(sele)),
                "mannwhitney_sel_vs_stab_p": mannwhitney_p(sele, stab, rng),
            }
        )
        for r, tv in zip(pairs, t_pairs):
            pair_rows.append(
                {
                    "module": name,
                    "type_a": r["type_a"],
                    "type_b": r["type_b"],
                    "tag": r["tag"],
                    "D_module": float(tv),
                    "mean_S": (r["S_long_pulse"] + r["S_ramp"] + r["S_brief_pulse"]) / 3.0,
                    "rank_range": r["rank_range"],
                }
            )
    write_csv(out / "q3c_module_tests.csv", score_rows)
    write_csv(out / "q3c_pairs_with_D.csv", pair_rows)

    # Joint excitability-module geometry (exclude marker controls)
    excit_idx = [j for j, n in enumerate(module_names) if n not in MARKER_MODULES]
    joint = {}
    if len(excit_idx) >= 2:
        Mex = M[:, excit_idx]
        D = pairwise_euclid(Mex)
        t_pairs = pair_T(types, D, pairs)
        rho_s = spearman(t_pairs, mean_s)
        rho_r = spearman(t_pairs, rank_range)
        stab = t_pairs[tags == "stable"]
        sele = t_pairs[tags == "selective"]
        joint = {
            "n_modules": len(excit_idx),
            "spearman_T_vs_meanS": rho_s,
            "empirical_p_T_vs_meanS": empirical_spearman_p(t_pairs, mean_s, rng, N_SHUFFLE),
            "spearman_T_vs_rank_range": rho_r,
            "empirical_p_T_vs_rank_range": empirical_spearman_p(t_pairs, rank_range, rng, N_SHUFFLE),
            "mean_T_stable": float(np.nanmean(stab)),
            "mean_T_selective": float(np.nanmean(sele)),
            "mannwhitney_sel_vs_stab_p": mannwhitney_p(sele, stab, rng),
        }
        write_csv(
            out / "q3c_joint_excitability_pairs.csv",
            [
                {
                    "type_a": r["type_a"],
                    "type_b": r["type_b"],
                    "tag": r["tag"],
                    "T_excitability": float(tv),
                    "mean_S": (r["S_long_pulse"] + r["S_ramp"] + r["S_brief_pulse"]) / 3.0,
                    "rank_range": r["rank_range"],
                }
                for r, tv in zip(pairs, t_pairs)
            ],
        )

    # Landing: modules track shared backbone if |ρ(D,meanS)| >> |ρ(D,rank_range)|
    excit_rows = [r for r in score_rows if r["role"] == "excitability"]
    mean_rho_s = float(np.nanmean([r["spearman_D_vs_meanS"] for r in excit_rows])) if excit_rows else float("nan")
    mean_rho_r = float(np.nanmean([abs(r["spearman_D_vs_rank_range"]) for r in excit_rows])) if excit_rows else float("nan")
    if joint.get("spearman_T_vs_meanS", 0) >= 0.25 and abs(joint.get("spearman_T_vs_rank_range", 0)) < 0.15:
        landing = "modules_track_shared_backbone"
    elif any(abs(r["spearman_D_vs_rank_range"]) >= 0.25 and r["empirical_p_D_vs_rank_range"] < 0.05 for r in excit_rows):
        landing = "some_modules_track_selectivity"
    else:
        landing = "modules_weak_or_mixed"
    payload = {
        "question": "Q3c",
        "modules_used": used,
        "module_tests": score_rows,
        "joint_excitability": joint,
        "mean_excitability_rho_D_vs_meanS": mean_rho_s,
        "mean_excitability_abs_rho_D_vs_rank_range": mean_rho_r,
        "landing": landing,
        "note": "Pre-specified programs only. Marker modules are controls. Not genome-wide fishing.",
    }
    (out / "q3c_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def make_q3b_figures(out: Path, pairs: list[dict], primary: dict) -> None:
    tcol = "T_primary"
    rows = []
    with open(out / "q3b_pairs_with_T_primary.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    tags = [r["tag"] for r in rows]
    T = np.array([float(r[tcol]) for r in rows])
    mean_s = np.array(
        [(float(r["S_long_pulse"]) + float(r["S_ramp"]) + float(r["S_brief_pulse"])) / 3.0 for r in rows]
    )
    rr = np.array([float(r["rank_range"]) for r in rows])

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    order = ["stable", "intermediate", "selective"]
    data = [T[np.array(tags) == k] for k in order]
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_ylabel("RNA centroid distance T")
    ax.set_title("Q3b: T by Q3a pair tag")
    fig.tight_layout()
    fig.savefig(out / "q3b_T_by_tag.png", dpi=160)
    fig.savefig(out / "q3b_T_by_tag.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    axes[0].scatter(mean_s, T, s=8, alpha=0.35, c="#335588")
    axes[0].set_xlabel("mean phenotype S")
    axes[0].set_ylabel("RNA T")
    axes[0].set_title(f"T vs mean S  ρ={primary['spearman_T_vs_meanS']:.2f}")
    axes[1].scatter(rr, T, s=8, alpha=0.35, c="#884433")
    axes[1].set_xlabel("rank range (selectivity)")
    axes[1].set_ylabel("RNA T")
    axes[1].set_title(f"T vs rank range  ρ={primary['spearman_T_vs_rank_range']:.2f}")
    fig.tight_layout()
    fig.savefig(out / "q3b_T_vs_S_and_selectivity.png", dpi=160)
    fig.savefig(out / "q3b_T_vs_S_and_selectivity.pdf")
    plt.close(fig)

    with open(out / "q3b_type_selectivity_primary.csv", newline="") as f:
        trows = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    x = np.array([float(r["RNA_PC1"]) for r in trows])
    y = np.array([float(r["selectivity_index"]) for r in trows])
    ax.scatter(x, y, s=28, c="#226644")
    for r in trows:
        if abs(float(r["selectivity_index"])) >= 0.35:
            ax.text(float(r["RNA_PC1"]), float(r["selectivity_index"]), r["t_type"].split()[0], fontsize=6)
    ax.set_xlabel("RNA PC1 (type centroid)")
    ax.set_ylabel("selectivity index")
    ax.set_title("Q3b: RNA PC1 vs type selectivity")
    fig.tight_layout()
    fig.savefig(out / "q3b_type_rna_pc_vs_selectivity.png", dpi=160)
    fig.savefig(out / "q3b_type_rna_pc_vs_selectivity.pdf")
    plt.close(fig)

    gpath = out / "q3b_genes_vs_selectivity_primary.csv"
    if gpath.exists():
        with open(gpath, newline="") as f:
            grows = list(csv.DictReader(f))[:18]
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        ys = np.arange(len(grows))
        vals = [float(r["spearman"]) for r in grows]
        ax.barh(ys, vals, color=["#335588" if v > 0 else "#aa5533" for v in vals])
        ax.set_yticks(ys)
        ax.set_yticklabels([r["gene"] for r in grows], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Spearman(type expression, selectivity index)")
        ax.set_title("Q3b: top panel genes vs selectivity (FDR within panel)")
        fig.tight_layout()
        fig.savefig(out / "q3b_top_genes_selectivity.png", dpi=160)
        fig.savefig(out / "q3b_top_genes_selectivity.pdf")
        plt.close(fig)


def make_q3c_figures(out: Path, payload: dict) -> None:
    rows = payload["module_tests"]
    names = [r["module"] for r in rows]
    rho_s = [r["spearman_D_vs_meanS"] for r in rows]
    rho_r = [r["spearman_D_vs_rank_range"] for r in rows]
    colors = ["#888888" if r["role"] == "marker_control" else "#226688" for r in rows]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    x = np.arange(len(names))
    ax.bar(x - 0.18, rho_s, 0.36, label="ρ(D, mean S)", color=colors)
    ax.bar(x + 0.18, rho_r, 0.36, label="ρ(D, rank range)", color="#cc8844")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Spearman")
    ax.set_title(f"Q3c: module distances — {payload['landing']}")
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "q3c_module_D_concordance.png", dpi=160)
    fig.savefig(out / "q3c_module_D_concordance.pdf")
    plt.close(fig)

    with open(out / "q3c_type_module_scores.csv", newline="") as f:
        trows = list(csv.DictReader(f))
    mods = [c for c in trows[0] if c not in {"t_type", "subclass", "selectivity_index"}]
    mat = np.array([[float(r[m]) for m in mods] for r in trows], dtype=float)
    # z-score columns for display
    mat = (mat - mat.mean(axis=0)) / np.where(mat.std(axis=0) == 0, 1.0, mat.std(axis=0))
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-2.2, vmax=2.2)
    ax.set_xticks(range(len(mods)))
    ax.set_xticklabels(mods, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(trows)))
    ax.set_yticklabels([r["t_type"] if len(r["t_type"]) < 22 else r["t_type"][:20] + "…" for r in trows], fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.6, label="z module score")
    ax.set_title("Q3c: t-type scores on pre-specified programs")
    fig.tight_layout()
    fig.savefig(out / "q3c_type_module_heatmap.png", dpi=160)
    fig.savefig(out / "q3c_type_module_heatmap.pdf")
    plt.close(fig)


def main() -> None:
    OUT_B.mkdir(parents=True, exist_ok=True)
    OUT_C.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    curated_wanted = read_gene_list(GENE_LIST)
    sample_to_type = load_join(JOIN)
    pairs = load_pairs(PAIRS)
    types = sorted(set(r["type_a"] for r in pairs) | set(r["type_b"] for r in pairs))
    type_set = set(types)
    keep = sorted(sid for sid, tt in sample_to_type.items() if tt in type_set)
    print(f"Q3b/c: {len(types)} types, {len(keep)} RNA samples to load", flush=True)

    genes, counts, samples = load_counts(nemo_counts_csv(), keep)
    labels = [sample_to_type[s] for s in samples]
    print(f"Loaded counts {counts.shape} (genes x cells)", flush=True)
    expr_all = log_cpm(counts)

    gset = {g: i for i, g in enumerate(genes)}
    curated = [g for g in curated_wanted if g in gset]
    var = expr_all.var(axis=1)
    hvg_order = np.argsort(-var)
    hvg = [genes[i] for i in hvg_order[:N_HVG]]
    primary = list(curated)
    for g in hvg:
        if g not in primary:
            primary.append(g)
        if len(primary) >= N_PRIMARY:
            break

    sel = type_selectivity(types, pairs)
    panels = {
        "curated": curated,
        "primary": primary,
        "hvg500": hvg,
    }
    summaries = {}
    for name, panel in panels.items():
        idx = np.array([gset[g] for g in panel], dtype=int)
        print(f"Q3b panel {name}: {len(panel)} genes", flush=True)
        summaries[name] = run_panel(
            name,
            expr_all[idx],
            samples,
            labels,
            types,
            pairs,
            sel,
            panel,
            rng,
            OUT_B,
            gene_table=name in {"curated", "primary"},
        )

    prim = summaries["primary"]
    if prim["spearman_T_vs_meanS"] >= 0.25 and abs(prim["spearman_T_vs_rank_range"]) < 0.15:
        landing = "rna_tracks_phenotype_distance"
    elif abs(prim["spearman_T_vs_rank_range"]) >= 0.25:
        landing = "rna_tracks_selectivity"
    else:
        landing = "rna_weak"
    q3b = {
        "question": "Q3b",
        "n_types": len(types),
        "types": types,
        "panels": summaries,
        "landing": landing,
        "rationale": (
            f"ρ(T,meanS)={prim['spearman_T_vs_meanS']:.2f} (emp_p={prim['empirical_p_T_vs_meanS']:.3f}); "
            f"ρ(T,rank_range)={prim['spearman_T_vs_rank_range']:.2f} (emp_p={prim['empirical_p_T_vs_rank_range']:.3f}); "
            f"meanT stable={prim['mean_T_stable']:.2f} vs selective={prim['mean_T_selective']:.2f}."
        ),
        "note": "Targeted curated∪HVG analysis — not a genome-wide discovery claim.",
    }
    (OUT_B / "q3b_summary.json").write_text(json.dumps(q3b, indent=2) + "\n")
    write_csv(OUT_B / "q3b_panel_summary.csv", list(summaries.values()))
    make_q3b_figures(OUT_B, pairs, prim)
    (OUT_B / "README.md").write_text(
        "\n".join(
            [
                "# Q3b — transcriptomic dimensions",
                "",
                "**Question:** Which transcriptomic axes distinguish types involved in stable vs selective (Q3a) relationships?",
                "",
                f"**Landing:** `{landing}`",
                "",
                q3b["rationale"],
                "",
                "## Design (targeted)",
                "",
                "- NeMO Patch-seq counts (`20200513_Mouse_PatchSeq_Release_count.v2`)",
                "- Join via `transcriptomics_sample_id`",
                "- Gene sets: curated excitability/markers · primary = curated ∪ top HVGs · HVG500 sensitivity",
                "- RNA PCA → type-centroid distances `T(a,b)` → relate to Q3a pair tags",
                "",
            ]
        )
        + "\n"
    )

    print("Q3c modules…", flush=True)
    curated_idx = np.array([gset[g] for g in curated], dtype=int)
    q3c = q3c_modules(curated, expr_all[curated_idx], labels, types, pairs, sel, rng, OUT_C)
    make_q3c_figures(OUT_C, q3c)

    lines = [
        "# Q3b/c results",
        "",
        f"## Q3b landing: **{landing}**",
        q3b["rationale"],
        "",
        "### Panels",
    ]
    for name, s in summaries.items():
        lines.append(
            f"- {name}: n_genes={s['n_genes']}, ρ(T,meanS)={s['spearman_T_vs_meanS']:.3f} "
            f"(emp_p={s['empirical_p_T_vs_meanS']:.3f}), ρ(T,rank_range)={s['spearman_T_vs_rank_range']:.3f} "
            f"(emp_p={s['empirical_p_T_vs_rank_range']:.3f})"
        )
    lines += ["", f"## Q3c landing: **{q3c['landing']}**", "Pre-specified programs only.", "", "### Module tests"]
    for r in q3c["module_tests"]:
        lines.append(
            f"- {r['module']} [{r['role']}]: ρ(D,meanS)={r['spearman_D_vs_meanS']:.3f}, "
            f"ρ(D,rank_range)={r['spearman_D_vs_rank_range']:.3f}, "
            f"ρ(score,selectivity)={r['spearman_score_vs_selectivity']:.3f}"
        )
    if q3c.get("joint_excitability"):
        j = q3c["joint_excitability"]
        lines += [
            "",
            "### Joint excitability-module geometry",
            f"- ρ(T,meanS)={j['spearman_T_vs_meanS']:.3f} (emp_p={j['empirical_p_T_vs_meanS']:.3f})",
            f"- ρ(T,rank_range)={j['spearman_T_vs_rank_range']:.3f} (emp_p={j['empirical_p_T_vs_rank_range']:.3f})",
        ]
    (OUT_C / "q3c_landing.md").write_text("\n".join(lines) + "\n")
    (OUT_B / "q3b_landing.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {OUT_B} and {OUT_C}")


if __name__ == "__main__":
    main()
