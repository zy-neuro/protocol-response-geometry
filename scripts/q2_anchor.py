#!/usr/bin/env python3
"""Q2 ANCHOR: is cellular phenotype organization biologically anchored?

Primary: A_p = Spearman(D^mol, D^phenotype).
Control: Cre alignment and within-Cre / Cre-residual t-type tests.
No S_p model comparison here (that is Q3).
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PHENO = ROOT / "results/phenotype"
JOIN = ROOT / "results/m0/m0_molecular_join.csv"
OUT = ROOT / "results/q2"

SEED = 0
N_SHUFFLE = 100
KNN = 15
VARIANCE_FLOOR = 0.80
N_PAIR_SAMPLE = 80000
MIN_CRE_FOR_WITHIN = 80
MIN_TTYPES_IN_CRE = 2

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


def choose_k(path: Path, n_available: int) -> int:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    k = n_available
    for row in rows:
        try:
            if float(row["cumulative_explained_variance_ratio"]) >= VARIANCE_FLOOR:
                k = int(float(row["component"]))
                break
        except (KeyError, ValueError):
            continue
    return max(1, min(k, n_available))


def load_scores(path: Path) -> tuple[list[str], np.ndarray]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = pc_cols(reader.fieldnames or [])
        stems, mats = [], []
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


def load_join(path: Path) -> dict[str, dict]:
    with open(path, newline="") as f:
        return {row["file_stem"]: row for row in csv.DictReader(f)}


def pairwise_euclidean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    nrm = np.sum(x * x, axis=1, keepdims=True)
    d2 = nrm + nrm.T - 2.0 * np.dot(x, x.T)
    np.maximum(d2, 0.0, out=d2)
    d = np.sqrt(d2)
    np.fill_diagonal(d, 0.0)
    return d


def sample_pairs(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    max_pairs = n * (n - 1) // 2
    take = min(N_PAIR_SAMPLE, max_pairs)
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
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra.astype(float), rb.astype(float))[0, 1])


def mol_distance(labels: np.ndarray, kind: str, extra: np.ndarray | None = None) -> np.ndarray:
    """Discrete molecular / genetic distances on a sampled pair set later; here full n×n."""
    n = labels.size
    if kind == "binary":
        return (labels[:, None] != labels[None, :]).astype(np.float64)
    if kind == "hier":
        # 0 same t-type, 1 same subclass else, 2 otherwise
        assert extra is not None
        same_t = labels[:, None] == labels[None, :]
        same_s = extra[:, None] == extra[None, :]
        d = np.where(same_t, 0.0, np.where(same_s, 1.0, 2.0))
        return d
    raise ValueError(kind)


def align_D(d_pheno: np.ndarray, d_mol: np.ndarray, ii: np.ndarray, jj: np.ndarray, rng: np.random.Generator, labels: np.ndarray) -> dict:
    ua, ub = d_pheno[ii, jj], d_mol[ii, jj]
    rho = spearman(ua, ub)
    null = np.empty(N_SHUFFLE, dtype=float)
    n = d_pheno.shape[0]
    for t in range(N_SHUFFLE):
        perm = rng.permutation(n)
        # shuffle molecular labels via permuting D_mol rows/cols equivalently
        null[t] = spearman(ua, d_mol[perm[ii], perm[jj]])
    p = float((1.0 + np.sum(null >= rho)) / (1.0 + N_SHUFFLE))
    return {
        "A_spearman": float(rho),
        "null_mean": float(np.mean(null)),
        "null_sd": float(np.std(null)),
        "empirical_p": p,
        "n_pairs": int(ii.size),
        "n_cells": int(n),
        "n_labels": int(len(set(labels.tolist()))),
    }


def eta2_multivariate(x: np.ndarray, labels: np.ndarray) -> dict:
    """Between-group SS / total SS in phenotype space."""
    x = np.asarray(x, dtype=np.float64)
    grand = x.mean(axis=0)
    total = np.sum((x - grand) ** 2)
    between = 0.0
    counts = Counter(labels.tolist())
    for lab, n in counts.items():
        mu = x[labels == lab].mean(axis=0)
        between += n * float(np.sum((mu - grand) ** 2))
    eta = float(between / total) if total > 0 else float("nan")
    return {
        "eta2": eta,
        "n_groups": int(len(counts)),
        "n_cells": int(x.shape[0]),
    }


def knn_same_label(d: np.ndarray, labels: np.ndarray, k: int = KNN) -> dict:
    n = d.shape[0]
    k = min(k, n - 1)
    same = np.empty(n, dtype=float)
    chance = np.empty(n, dtype=float)
    counts = Counter(labels.tolist())
    for i in range(n):
        order = np.argpartition(d[i], k + 1)
        neigh = [j for j in order[: k + 1] if j != i][:k]
        if len(neigh) < k:
            neigh = [j for j in np.argsort(d[i]) if j != i][:k]
        same[i] = np.mean([labels[j] == labels[i] for j in neigh])
        chance[i] = (counts[labels[i]] - 1) / (n - 1)
    return {
        "knn_same_mean": float(np.mean(same)),
        "knn_chance_mean": float(np.mean(chance)),
        "knn_enrichment": float(np.mean(same) / np.mean(chance)) if np.mean(chance) else float("nan"),
    }


def residualize_by_group(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.array(x, dtype=np.float64, copy=True)
    for g in set(groups.tolist()):
        idx = groups == g
        out[idx] -= out[idx].mean(axis=0)
    return out


def encode(values: list[str]) -> np.ndarray:
    uniq = {v: i for i, v in enumerate(sorted(set(values)))}
    return np.array([uniq[v] for v in values], dtype=int)


def protocol_anchor(
    proto: str,
    stems: list[str],
    X: np.ndarray,
    k: int,
    join: dict[str, dict],
    rng: np.random.Generator,
) -> tuple[dict, list[dict]]:
    recs = []
    keep_t, keep_c = [], []
    for i, fs in enumerate(stems):
        j = join.get(fs)
        if not j:
            continue
        t = (j.get("t_type") or "").strip()
        c = (j.get("cre_line") or "").strip()
        sc = (j.get("subclass") or "").strip()
        if t:
            keep_t.append((i, t, sc, c))
        if c:
            keep_c.append((i, c))

    xi = np.array([i for i, *_ in keep_t], dtype=int)
    x = X[xi, :k]
    t_labels = encode([t for _, t, *_ in keep_t])
    subclasses = encode([sc for _, _, sc, _ in keep_t])
    cres_for_t = [c for *_, c in keep_t]
    d_pheno = pairwise_euclidean(x)
    ii, jj = sample_pairs(x.shape[0], rng)

    d_t = mol_distance(t_labels, "binary")
    d_h = mol_distance(t_labels, "hier", subclasses)
    a_t = align_D(d_pheno, d_t, ii, jj, rng, t_labels)
    a_h = align_D(d_pheno, d_h, ii, jj, rng, t_labels)
    eta_t = eta2_multivariate(x, t_labels)
    knn_t = knn_same_label(d_pheno, t_labels)

    # Cre on overlapping labeled set (cells with both)
    cre_ok = [(i, t, sc, c) for i, t, sc, c in keep_t if c]
    xi_c = np.array([i for i, *_ in cre_ok], dtype=int)
    x_c = X[xi_c, :k]
    cre_lab = encode([c for *_, c in cre_ok])
    t_on_c = encode([t for _, t, *_ in cre_ok])
    d_pheno_c = pairwise_euclidean(x_c)
    ii_c, jj_c = sample_pairs(x_c.shape[0], rng)
    d_cre = mol_distance(cre_lab, "binary")
    a_cre = align_D(d_pheno_c, d_cre, ii_c, jj_c, rng, cre_lab)
    eta_cre = eta2_multivariate(x_c, cre_lab)

    # residualize Cre, re-test t-type
    x_res = residualize_by_group(x_c, cre_lab)
    d_res = pairwise_euclidean(x_res)
    d_t_c = mol_distance(t_on_c, "binary")
    a_res = align_D(d_res, d_t_c, ii_c, jj_c, rng, t_on_c)
    eta_res = eta2_multivariate(x_res, t_on_c)

    # within-Cre A_p
    within_rows = []
    by_cre = defaultdict(list)
    for idx_local, (_, t, sc, c) in enumerate(cre_ok):
        if c:
            by_cre[c].append(idx_local)
    for cre, idxs in by_cre.items():
        if len(idxs) < MIN_CRE_FOR_WITHIN:
            continue
        t_sub = t_on_c[np.array(idxs)]
        if len(set(t_sub.tolist())) < MIN_TTYPES_IN_CRE:
            continue
        xs = x_c[np.array(idxs)]
        ds = pairwise_euclidean(xs)
        iis, jjs = sample_pairs(xs.shape[0], rng)
        dt = mol_distance(t_sub, "binary")
        aw = align_D(ds, dt, iis, jjs, rng, t_sub)
        within_rows.append(
            {
                "protocol": proto,
                "cre_line": cre,
                "n_cells": aw["n_cells"],
                "n_t_types": aw["n_labels"],
                "A_spearman": aw["A_spearman"],
                "empirical_p": aw["empirical_p"],
                "eta2": eta2_multivariate(xs, t_sub)["eta2"],
            }
        )

    summary = {
        "protocol": proto,
        "k": int(k),
        "n_with_t_type": int(x.shape[0]),
        "n_t_types": a_t["n_labels"],
        "n_with_cre": int(x_c.shape[0]),
        "n_cre_lines": a_cre["n_labels"],
        "A_t_type": a_t["A_spearman"],
        "A_t_type_p": a_t["empirical_p"],
        "A_t_type_null": a_t["null_mean"],
        "A_hier": a_h["A_spearman"],
        "A_hier_p": a_h["empirical_p"],
        "A_cre": a_cre["A_spearman"],
        "A_cre_p": a_cre["empirical_p"],
        "A_t_type_after_cre_residual": a_res["A_spearman"],
        "A_t_type_after_cre_residual_p": a_res["empirical_p"],
        "eta2_t_type": eta_t["eta2"],
        "eta2_cre": eta_cre["eta2"],
        "eta2_t_type_after_cre_residual": eta_res["eta2"],
        "knn_t_type_same": knn_t["knn_same_mean"],
        "knn_t_type_chance": knn_t["knn_chance_mean"],
        "knn_t_type_enrichment": knn_t["knn_enrichment"],
        "n_cres_within_tested": len(within_rows),
        "within_cre_A_median": float(np.median([r["A_spearman"] for r in within_rows]))
        if within_rows
        else float("nan"),
    }
    recs.extend(within_rows)
    return summary, recs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    join = load_join(JOIN)

    summaries = []
    within_all = []
    for proto, path in SCORE_FILES.items():
        stems, X = load_scores(path)
        k = choose_k(VAR_FILES[proto], X.shape[1])
        print(f"Q2 {proto}: n={len(stems)} k={k}", flush=True)
        summary, within_rows = protocol_anchor(proto, stems, X, k, join, rng)
        summaries.append(summary)
        within_all.extend(within_rows)

    with open(OUT / "q2_anchor.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)
    if within_all:
        with open(OUT / "q2_within_cre.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(within_all[0].keys()))
            w.writeheader()
            w.writerows(within_all)

    payload = {
        "question": "Q2",
        "primary": "A_p = Spearman(D_mol_ttype, D_phenotype)",
        "control": "A_cre and A_t_type after Cre residual; within-Cre A_p",
        "seed": SEED,
        "n_shuffle": N_SHUFFLE,
        "protocols": summaries,
    }
    (OUT / "q2_summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    # figure: A_p t-type vs Cre vs residual
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    x = np.arange(len(summaries))
    width = 0.25
    ax.bar(x - width, [s["A_t_type"] for s in summaries], width, label="A_p t-type")
    ax.bar(x, [s["A_cre"] for s in summaries], width, label="A_p Cre (control)")
    ax.bar(
        x + width,
        [s["A_t_type_after_cre_residual"] for s in summaries],
        width,
        label="A_p t-type | Cre residual",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[s["protocol"]] for s in summaries])
    ax.set_ylabel("Spearman alignment A_p")
    ax.set_title("Q2: Biological anchoring of phenotype structure")
    ax.legend(frameon=False, fontsize=8)
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "q2_alignment.pdf")
    fig.savefig(OUT / "q2_alignment.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.bar(x - 0.16, [s["eta2_t_type"] for s in summaries], 0.32, label="η² t-type")
    ax.bar(x + 0.16, [s["eta2_cre"] for s in summaries], 0.32, label="η² Cre")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[s["protocol"]] for s in summaries])
    ax.set_ylabel("Variance explained (η²)")
    ax.set_title("Q2: Phenotype variance partitioned by labels")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "q2_eta2.pdf")
    fig.savefig(OUT / "q2_eta2.png", dpi=160)
    plt.close(fig)

    lines = [
        "# Q2 results — biological anchoring",
        "",
        "Primary A_p: Spearman(D^t-type binary, D^phenotype). Cre = control.",
        "",
    ]
    for s in summaries:
        lines.append(
            f"- {DISPLAY[s['protocol']]}: A_t={s['A_t_type']:.3f} (p={s['A_t_type_p']:.3f}, "
            f"null {s['A_t_type_null']:.3f}); A_Cre={s['A_cre']:.3f}; "
            f"A_t|Cre-residual={s['A_t_type_after_cre_residual']:.3f} "
            f"(p={s['A_t_type_after_cre_residual_p']:.3f}); "
            f"η²_t={s['eta2_t_type']:.3f}, η²_Cre={s['eta2_cre']:.3f}; "
            f"kNN enrich={s['knn_t_type_enrichment']:.2f}; "
            f"within-Cre median A={s['within_cre_A_median']:.3f} "
            f"(n Cres={s['n_cres_within_tested']})"
        )
    lines += [
        "",
        "Q2 asks whether, not which relationships are shared/selective (Q3).",
        "No preset that Brief must diverge.",
    ]
    (OUT / "q2_landing.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
