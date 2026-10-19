#!/usr/bin/env python3
"""Q3a EXPLAIN: shared vs selective t-type relationships.

Canonical script (k80, same as Q1/Q2). Writes results/q3a/.
S_p = Euclidean distance of t-type centroids.
Tags use rank-normalized S across the three regimes.
Model call is not Brief-centric.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PHENO = ROOT / "results/phenotype"
JOIN = ROOT / "results/m0/m0_molecular_join.csv"
OUT = ROOT / "results/q3a"

SEED = 0
N_SHUFFLE = 100
VARIANCE_FLOOR = 0.80
N_MIN = 10
STABLE_RANK_CV = 0.25
SELECTIVE_RANK_RANGE = 0.55

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
PROTOCOLS = ["long_pulse", "ramp", "brief_pulse"]


def pc_cols(fieldnames):
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


def load_scores(path: Path):
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


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def protocol_arrays(stems, X, k, join, label_key: str):
    xs, labs = [], []
    for i, fs in enumerate(stems):
        j = join.get(fs)
        if not j:
            continue
        lab = (j.get(label_key) or "").strip()
        if not lab:
            continue
        xs.append(X[i, :k])
        labs.append(lab)
    return np.asarray(xs, dtype=float), labs


def centroids(x, labels, types):
    mus, ns = [], {}
    for t in types:
        idx = np.array([i for i, lab in enumerate(labels) if lab == t], dtype=int)
        mus.append(x[idx].mean(axis=0) if idx.size else np.full(x.shape[1], np.nan))
        ns[t] = int(idx.size)
    M = np.asarray(mus, dtype=float)
    nrm = np.sum(M * M, axis=1, keepdims=True)
    d2 = nrm + nrm.T - 2.0 * (M @ M.T)
    np.maximum(d2, 0.0, out=d2)
    S = np.sqrt(d2)
    np.fill_diagonal(S, 0.0)
    return S, ns


def upper(S):
    i, j = np.triu_indices(S.shape[0], k=1)
    return S[i, j]


def call_model(rhos: list[float]) -> str:
    vals = np.array(rhos, dtype=float)
    mx, mn = float(np.max(vals)), float(np.min(vals))
    if mn >= 0.40 and (mx - mn) < 0.25:
        return "H1_shared"
    if mx >= 0.40 and (mx - mn) >= 0.25:
        return "H2_partial"
    if mx >= 0.20 and mn < 0.20:
        return "H2_partial"
    return "H3_regime_specific"


def rank_norm(v: np.ndarray) -> np.ndarray:
    r = np.argsort(np.argsort(v)).astype(float)
    return r / max(1.0, r.size - 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    join = load_join(JOIN)

    packed = {}
    for proto, path in SCORE_FILES.items():
        stems, X = load_scores(path)
        k = choose_k(VAR_FILES[proto], X.shape[1])
        x, labs = protocol_arrays(stems, X, k, join, "t_type")
        counts = Counter(labs)
        packed[proto] = {"k": k, "x": x, "labs": labs, "counts": counts}
        print(f"Q3a {proto}: k={k} n={len(labs)} types>={N_MIN}: {sum(n >= N_MIN for n in counts.values())}", flush=True)

    triple_types = sorted(
        t
        for t in set(packed["long_pulse"]["counts"])
        if all(packed[p]["counts"][t] >= N_MIN for p in PROTOCOLS)
    )
    S = {}
    ns = {}
    for p in PROTOCOLS:
        S[p], ns[p] = centroids(packed[p]["x"], packed[p]["labs"], triple_types)
        with open(OUT / f"q3a_S_{p}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_type_a", "t_type_b", "S", "n_a", "n_b"])
            for i, a in enumerate(triple_types):
                for j, b in enumerate(triple_types):
                    if i < j:
                        w.writerow([a, b, float(S[p][i, j]), ns[p][a], ns[p][b]])

    pairs = [("long_pulse", "ramp"), ("long_pulse", "brief_pulse"), ("ramp", "brief_pulse")]
    pair_rows = []
    for a, b in pairs:
        rho = spearman(upper(S[a]), upper(S[b]))
        null = []
        for _ in range(N_SHUFFLE):
            shuf = list(rng.permutation(np.array(packed[b]["labs"])))
            S_sh, _ = centroids(packed[b]["x"], shuf, triple_types)
            if np.isnan(S_sh).any():
                continue
            null.append(spearman(upper(S[a]), upper(S_sh)))
        null = np.array(null, dtype=float) if null else np.array([np.nan])
        p = float((1.0 + np.sum(null >= rho)) / (1.0 + null.size)) if np.isfinite(rho) else float("nan")
        pair_rows.append(
            {
                "protocol_a": a,
                "protocol_b": b,
                "n_types": len(triple_types),
                "n_type_pairs": int(len(triple_types) * (len(triple_types) - 1) / 2),
                "spearman_rho": rho,
                "null_mean": float(np.nanmean(null)),
                "empirical_p_vs_shuffle": p,
            }
        )

    rhos = [r["spearman_rho"] for r in pair_rows]
    model = call_model(rhos)

    # pair tags on rank-normalized S
    ii, jj = np.triu_indices(len(triple_types), k=1)
    ranks = {p: rank_norm(upper(S[p])) for p in PROTOCOLS}
    pair_calls = []
    for k, (i, j) in enumerate(zip(ii, jj)):
        rn = {p: float(ranks[p][k]) for p in PROTOCOLS}
        raw = {p: float(S[p][i, j]) for p in PROTOCOLS}
        vals = np.array(list(rn.values()))
        mean = float(vals.mean())
        cv = float(vals.std() / mean) if mean > 1e-9 else 0.0
        rng_r = float(vals.max() - vals.min())
        if cv < STABLE_RANK_CV:
            tag = "stable"
            driver = ""
        elif rng_r >= SELECTIVE_RANK_RANGE:
            tag = "selective"
            driver = max(PROTOCOLS, key=lambda p: abs(rn[p] - mean))
        else:
            tag = "intermediate"
            driver = ""
        pair_calls.append(
            {
                "type_a": triple_types[i],
                "type_b": triple_types[j],
                "subclass_a": triple_types[i].split()[0],
                "subclass_b": triple_types[j].split()[0],
                "same_subclass": str(triple_types[i].split()[0] == triple_types[j].split()[0]),
                "S_long_pulse": raw["long_pulse"],
                "S_ramp": raw["ramp"],
                "S_brief_pulse": raw["brief_pulse"],
                "rank_norm_long_pulse": rn["long_pulse"],
                "rank_norm_ramp": rn["ramp"],
                "rank_norm_brief_pulse": rn["brief_pulse"],
                "rank_mean": mean,
                "rank_cv": cv,
                "rank_range": rng_r,
                "tag": tag,
                "selective_driver": driver,
            }
        )

    tag_counts = Counter(r["tag"] for r in pair_calls)
    with open(OUT / "q3a_S_concordance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        w.writerows(pair_rows)
    with open(OUT / "q3a_pair_tags.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_calls[0].keys()))
        w.writeheader()
        w.writerows(pair_calls)

    # subclass sensitivity
    sub_S = {}
    for p in PROTOCOLS:
        stems, X = load_scores(SCORE_FILES[p])
        x, labs = protocol_arrays(stems, X, packed[p]["k"], join, "subclass")
        types = sorted(t for t, n in Counter(labs).items() if n >= N_MIN)
        sub_S[p], _ = centroids(x, labs, types)
        packed[p]["sub_types"] = types
    sub_rows = []
    for a, b in pairs:
        shared = sorted(set(packed[a]["sub_types"]) & set(packed[b]["sub_types"]))
        ia = [packed[a]["sub_types"].index(t) for t in shared]
        ib = [packed[b]["sub_types"].index(t) for t in shared]
        rho = spearman(upper(sub_S[a][np.ix_(ia, ia)]), upper(sub_S[b][np.ix_(ib, ib)]))
        sub_rows.append({"pair": f"{a}__{b}", "n_shared_subclasses": len(shared), "spearman_S": rho})
    with open(OUT / "q3a_subclass_concordance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sub_rows[0].keys()))
        w.writeheader()
        w.writerows(sub_rows)

    with open(OUT / "q3a_type_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_type"] + [f"n_{p}" for p in PROTOCOLS])
        for t in triple_types:
            w.writerow([t] + [packed[p]["counts"][t] for p in PROTOCOLS])

    payload = {
        "question": "Q3a",
        "k_rule": "variance_80",
        "k": {p: packed[p]["k"] for p in PROTOCOLS},
        "n_min": N_MIN,
        "n_shared_types": len(triple_types),
        "shared_types": triple_types,
        "n_type_pairs": len(pair_calls),
        "concordance": pair_rows,
        "subclass_concordance": sub_rows,
        "model_call": {
            "model": model,
            "mean_rho": float(np.mean(rhos)),
            "min_rho": float(np.min(rhos)),
            "max_rho": float(np.max(rhos)),
            "rule": "H1 if min>=0.40 and range<0.25; H2 if max>=0.40 and range>=0.25 or mixed; else H3",
        },
        "pair_tag_counts": dict(tag_counts),
        "thresholds": {
            "stable_rank_cv": STABLE_RANK_CV,
            "selective_rank_range": SELECTIVE_RANK_RANGE,
        },
    }
    (OUT / "q3a_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    with open(OUT / "q3a_model_call.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(payload["model_call"].keys()))
        w.writeheader()
        w.writerow(payload["model_call"])

    print(f"Model {model}  types={len(triple_types)}  tags={dict(tag_counts)}")
    for r in pair_rows:
        print(f"  {r['protocol_a']} vs {r['protocol_b']}: ρ={r['spearman_rho']:.3f} p={r['empirical_p_vs_shuffle']:.3f}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
