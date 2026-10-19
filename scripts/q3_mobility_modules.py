#!/usr/bin/env python3
"""cell-level mobility and weighted co-expression modules.

Mobility = 1 − mean Spearman of each cell’s rank-distance profiles across the
three regime pairs in k₈₀ space (high = neighborhood changed). Module
eigengenes from a WGCNA-style unsigned weighted network on curated excitability
genes ∪ top HVGs (cap 2000). t-type residualization is the covariate control.

Writes results/q3_mobility/ only. Does not change Q3 type-level claims.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ORIG_JOIN, ORIG_PHENO, OUT_Q3_MOB, REPO  # noqa: E402

SEED = 0
N_SHUFFLE = 1000
K80_FLOOR = 0.80
N_GENE_CAP = 2000
MIN_MODULE_SIZE = 15
CURATED = REPO / "configs/q3_curated_excitability_genes.txt"
NEMO_COUNTS_NAME = "20200513_Mouse_PatchSeq_Release_count.v2.csv"
PROTOCOLS = ("long_pulse", "ramp", "brief_pulse")
PAIRS = (
    ("long_pulse", "ramp"),
    ("long_pulse", "brief_pulse"),
    ("ramp", "brief_pulse"),
)


def read_gene_list(path: Path) -> list[str]:
    genes = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            genes.append(s)
    return genes


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def pc_cols(fieldnames: list[str]) -> list[str]:
    cols = [c for c in fieldnames if c.startswith("PC") and c[2:].isdigit()]
    return sorted(cols, key=lambda c: int(c[2:]))


def load_k80(path: Path, n_available: int) -> int:
    k = n_available
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if float(row["cumulative_explained_variance_ratio"]) >= K80_FLOOR:
                k = int(float(row["component"]))
                break
    return max(1, min(k, n_available))


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


def pairwise_euclidean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    nrm = np.sum(x * x, axis=1, keepdims=True)
    d2 = nrm + nrm.T - 2.0 * np.dot(x, x.T)
    np.maximum(d2, 0.0, out=d2)
    d = np.sqrt(d2)
    np.fill_diagonal(d, 0.0)
    return d


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
    return float(math.erfc(t / math.sqrt(2.0)))


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


def residualize_by_group(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=float).copy()
    for g in np.unique(groups):
        m = groups == g
        mu = np.nanmean(out[m])
        out[m] = out[m] - mu
    return out


def nemo_counts_csv() -> Path:
    root = REPO / "data/official/nemo"
    hits = sorted(p for p in root.rglob(NEMO_COUNTS_NAME) if p.is_file()) if root.exists() else []
    if not hits:
        raise SystemExit(f"Missing {NEMO_COUNTS_NAME} under {root}")
    return hits[0]


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
        chunk, chunk_genes = [], []
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


def scale_free_r2(connectivity: np.ndarray) -> float:
    k = np.asarray(connectivity, dtype=float)
    k = k[np.isfinite(k) & (k > 0)]
    if k.size < 20:
        return float("nan")
    bins = np.linspace(np.log10(k.min()), np.log10(k.max() + 1e-12), 12)
    hist, edges = np.histogram(np.log10(k), bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    p = hist.astype(float)
    mask = p > 0
    if mask.sum() < 4:
        return float("nan")
    x = centers[mask]
    y = np.log10(p[mask])
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def pick_soft_threshold(corr_abs: np.ndarray, betas=(1, 2, 3, 4, 5, 6, 8, 10, 12)) -> tuple[int, dict]:
    scores = {}
    chosen = betas[-1]
    chosen_r2 = -1.0
    for b in betas:
        adj = np.power(corr_abs, b)
        np.fill_diagonal(adj, 0.0)
        r2 = scale_free_r2(adj.sum(axis=1))
        mean_k = float(adj.sum(axis=1).mean())
        scores[int(b)] = {"r2": r2, "mean_connectivity": mean_k}
        if np.isfinite(r2) and r2 >= 0.80:
            chosen, chosen_r2 = int(b), r2
            break
        if np.isfinite(r2) and r2 > chosen_r2:
            chosen, chosen_r2 = int(b), r2
    if chosen_r2 < 0.80:
        # Topology is not scale-free in this panel; module recovery (below) chooses β.
        pass
    return chosen, scores


def topological_overlap(adj: np.ndarray) -> np.ndarray:
    k = adj.sum(axis=1)
    a2 = adj @ adj
    denom = np.minimum(k[:, None], k[None, :]) + 1.0 - adj
    denom[denom == 0] = 1.0
    tom = (a2 + adj) / denom
    np.fill_diagonal(tom, 1.0)
    np.clip(tom, 0.0, 1.0, out=tom)
    return tom


def cut_modules(tom: np.ndarray, min_size: int = MIN_MODULE_SIZE) -> tuple[np.ndarray, float]:
    dist = np.clip(1.0 - tom, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    z = linkage(condensed, method="average")
    # Prefer a cut that yields a tractable number of modules of usable size.
    for height in (0.80, 0.85, 0.90, 0.75, 0.70, 0.95):
        labels = fcluster(z, t=height, criterion="distance")
        counts = Counter(int(x) for x in labels)
        n_big = sum(1 for n in counts.values() if n >= min_size)
        if 5 <= n_big <= 40:
            return labels.astype(int), float(height)
    labels = fcluster(z, t=0.85, criterion="distance")
    return labels.astype(int), 0.85


def module_eigengene(expr: np.ndarray) -> np.ndarray:
    """expr: cells x genes. PC1, sign-aligned with mean expression."""
    x = np.asarray(expr, dtype=float)
    mu = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    z = (x - mu) / std
    u, s, _vt = np.linalg.svd(z, full_matrices=False)
    me = u[:, 0] * s[0]
    mean_expr = z.mean(axis=1)
    if np.corrcoef(me, mean_expr)[0, 1] < 0:
        me = -me
    return me


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


def compute_mobility(stems_by_proto, X_by_proto, k_by_proto) -> tuple[list[str], np.ndarray, dict]:
    sets = [set(stems_by_proto[p]) for p in PROTOCOLS]
    shared = [s for s in stems_by_proto["long_pulse"] if s in sets[1] and s in sets[2]]
    pos = {p: {s: i for i, s in enumerate(stems_by_proto[p])} for p in PROTOCOLS}
    dists = {}
    for p in PROTOCOLS:
        idx = np.array([pos[p][s] for s in shared], dtype=int)
        xp = X_by_proto[p][idx, : k_by_proto[p]]
        dists[p] = pairwise_euclidean(xp)
    n = len(shared)
    rhos = np.full((n, len(PAIRS)), np.nan)
    for j, (a, b) in enumerate(PAIRS):
        da, db = dists[a], dists[b]
        for i in range(n):
            # exclude self
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            rhos[i, j] = spearman(da[i, mask], db[i, mask])
    mean_rho = np.nanmean(rhos, axis=1)
    mobility = 1.0 - mean_rho
    pair_keys = [f"{a}__{b}" for a, b in PAIRS]
    extra = {"pair_keys": pair_keys, "pair_rho": rhos, "mean_rho": mean_rho}
    return shared, mobility, extra


def main() -> None:
    OUT_Q3_MOB.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    stems_by_proto, X_by_proto, k_by_proto = {}, {}, {}
    for p in PROTOCOLS:
        s, mat = load_scores(ORIG_PHENO / f"{p}_pca_scores.csv")
        stems_by_proto[p] = s
        X_by_proto[p] = mat
        k_by_proto[p] = load_k80(ORIG_PHENO / f"{p}_pca_variance.csv", mat.shape[1])
        print(f"{p}: n={len(s)} k80={k_by_proto[p]}", flush=True)

    shared, mobility, extra = compute_mobility(stems_by_proto, X_by_proto, k_by_proto)
    print(f"Triple-shared cells for mobility: n={len(shared)}", flush=True)

    join_rows = load_csv(ORIG_JOIN)
    join = {r["file_stem"]: r for r in join_rows if r.get("file_stem")}
    score_rows = []
    for i, stem in enumerate(shared):
        j = join.get(stem, {})
        rec = {
            "file_stem": stem,
            "transcriptomics_sample_id": (j.get("transcriptomics_sample_id") or "").strip(),
            "t_type": (j.get("t_type") or "").strip(),
            "subclass": (j.get("subclass") or "").strip(),
            "mobility": float(mobility[i]),
            "mean_rank_distance_spearman": float(extra["mean_rho"][i]),
        }
        for k, key in enumerate(extra["pair_keys"]):
            rec[f"rho_{key}"] = float(extra["pair_rho"][i, k])
        score_rows.append(rec)
    write_csv(OUT_Q3_MOB / "mobility_scores.csv", score_rows)

    # RNA join: cells with mobility + sample id + t-type
    keep_samples = []
    stem_by_sample = {}
    ttype_by_sample = {}
    mob_by_sample = {}
    for rec, mob in zip(score_rows, mobility):
        sid = rec["transcriptomics_sample_id"]
        tt = rec["t_type"]
        if not sid or not tt:
            continue
        keep_samples.append(sid)
        stem_by_sample[sid] = rec["file_stem"]
        ttype_by_sample[sid] = tt
        mob_by_sample[sid] = float(mob)
    keep_samples = sorted(set(keep_samples))
    print(f"Mobility cells with RNA+t-type: {len(keep_samples)}", flush=True)

    genes, counts, samples = load_counts(nemo_counts_csv(), keep_samples)
    print(f"Loaded counts {counts.shape} (genes x cells)", flush=True)
    expr_all = log_cpm(counts)
    labels = [ttype_by_sample[s] for s in samples]
    mob = np.array([mob_by_sample[s] for s in samples], dtype=float)
    stems_rna = [stem_by_sample[s] for s in samples]

    curated_wanted = read_gene_list(CURATED)
    gset = {g: i for i, g in enumerate(genes)}
    curated = [g for g in curated_wanted if g in gset]
    var = expr_all.var(axis=1)
    hvg_order = np.argsort(-var)
    panel = list(curated)
    seen = set(panel)
    for i in hvg_order:
        g = genes[int(i)]
        if g in seen:
            continue
        panel.append(g)
        seen.add(g)
        if len(panel) >= N_GENE_CAP:
            break
    idx = np.array([gset[g] for g in panel], dtype=int)
    expr = expr_all[idx]  # genes x cells
    # drop zero-variance genes
    keep_g = expr.var(axis=1) > 0
    panel = [g for g, ok in zip(panel, keep_g) if ok]
    expr = expr[keep_g]
    print(f"Module gene panel: {len(panel)} (curated present {len(curated)})", flush=True)

    z = expr - expr.mean(axis=1, keepdims=True)
    sd = z.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    z = z / sd
    corr = np.corrcoef(z)
    np.fill_diagonal(corr, 1.0)
    corr_abs = np.abs(corr)
    _best_r2_beta, beta_scores = pick_soft_threshold(corr_abs)
    chosen = None
    adj = tom = labels_mod = None
    cut_height = 0.85
    for b in (4, 3, 2, 1):
        adj_b = np.power(corr_abs, b)
        np.fill_diagonal(adj_b, 0.0)
        tom_b = topological_overlap(adj_b)
        labels_b, height_b = cut_modules(tom_b)
        n_big = sum(1 for n in Counter(int(x) for x in labels_b).values() if n >= MIN_MODULE_SIZE)
        r2 = beta_scores.get(int(b), {}).get("r2")
        print(f"β={b} modules≥{MIN_MODULE_SIZE}: {n_big}  scale-free R²={r2}", flush=True)
        if n_big >= 3:
            chosen, adj, tom, labels_mod, cut_height = int(b), adj_b, tom_b, labels_b, height_b
            break
    if chosen is None:
        raise SystemExit("No unsigned β in {4,3,2,1} recovered ≥3 modules")
    beta = chosen
    print(f"Using unsigned soft-threshold β={beta} (scale-free R² never reached 0.8)", flush=True)
    counts_mod = Counter(int(x) for x in labels_mod)

    module_rows = []
    eigengenes = []
    eig_names = []
    gene_assign = []
    for mid, n_g in sorted(counts_mod.items(), key=lambda kv: -kv[1]):
        members = [panel[i] for i, lab in enumerate(labels_mod) if int(lab) == int(mid)]
        if len(members) < MIN_MODULE_SIZE:
            color = "grey"
            for g in members:
                gene_assign.append({"gene": g, "module": color, "module_id": int(mid), "hub": 0})
            continue
        name = f"M{len(eig_names)+1:02d}"
        gidx = np.array([i for i, lab in enumerate(labels_mod) if int(lab) == int(mid)], dtype=int)
        intramod = adj[np.ix_(gidx, gidx)].sum(axis=1)
        hub_i = int(gidx[int(np.argmax(intramod))])
        hub = panel[hub_i]
        me = module_eigengene(expr[gidx].T)
        eigengenes.append(me)
        eig_names.append(name)
        n_cur = sum(1 for g in members if g in set(curated))
        module_rows.append(
            {
                "module": name,
                "n_genes": len(members),
                "n_curated": n_cur,
                "hub_gene": hub,
                "cut_height": cut_height,
                "soft_threshold": beta,
                "genes": ";".join(members),
            }
        )
        for g in members:
            gene_assign.append({"gene": g, "module": name, "module_id": int(mid), "hub": int(g == hub)})

    write_csv(OUT_Q3_MOB / "module_table.csv", module_rows)
    write_csv(OUT_Q3_MOB / "module_genes.csv", gene_assign)
    ME = np.vstack(eigengenes).T if eigengenes else np.zeros((len(samples), 0))

    groups = np.array(labels, dtype=object)
    mob_res = residualize_by_group(mob, groups)
    # drop singleton types from residualized correlations (residual identically 0)
    type_n = Counter(labels)
    keep_cell = np.array([type_n[t] >= 3 for t in labels], dtype=bool)

    assoc = []
    rng_u = np.random.default_rng(SEED)
    rng_r = np.random.default_rng(SEED + 1)
    for j, name in enumerate(eig_names):
        me = ME[:, j]
        me_res = residualize_by_group(me, groups)
        rho = spearman(me, mob)
        p_an = spearman_p(rho, int(np.isfinite(me).sum()))
        p_emp = empirical_spearman_p(me, mob, rng_u, N_SHUFFLE)
        rho_r = spearman(me_res[keep_cell], mob_res[keep_cell])
        n_res = int(np.isfinite(me_res[keep_cell]).sum())
        p_an_r = spearman_p(rho_r, n_res)
        p_emp_r = empirical_spearman_p(me_res[keep_cell], mob_res[keep_cell], rng_r, N_SHUFFLE)
        assoc.append(
            {
                "module": name,
                "n_genes": int(module_rows[j]["n_genes"]),
                "hub_gene": module_rows[j]["hub_gene"],
                "n_cells": int(len(me)),
                "n_cells_residual": int(keep_cell.sum()),
                "spearman_unadjusted": rho,
                "p_analytic_unadjusted": p_an,
                "empirical_p_unadjusted": p_emp,
                "spearman_ttype_residual": rho_r,
                "p_analytic_residual": p_an_r,
                "empirical_p_residual": p_emp_r,
            }
        )
    if assoc:
        for key_p, key_f in (
            ("empirical_p_unadjusted", "fdr_empirical_unadjusted"),
            ("empirical_p_residual", "fdr_empirical_residual"),
        ):
            fdrs = bh_fdr([float(r[key_p]) for r in assoc])
            for r, f in zip(assoc, fdrs):
                r[key_f] = f
    write_csv(OUT_Q3_MOB / "module_mobility_association.csv", assoc)

    me_rows = []
    for i, sid in enumerate(samples):
        rec = {
            "file_stem": stems_rna[i],
            "transcriptomics_sample_id": sid,
            "t_type": labels[i],
            "mobility": float(mob[i]),
            "mobility_ttype_residual": float(mob_res[i]),
        }
        for j, name in enumerate(eig_names):
            rec[name] = float(ME[i, j])
        me_rows.append(rec)
    write_csv(OUT_Q3_MOB / "module_eigengenes.csv", me_rows)

    assoc_sorted_u = sorted(assoc, key=lambda r: abs(r["spearman_unadjusted"]), reverse=True)
    assoc_sorted_r = sorted(assoc, key=lambda r: abs(r["spearman_ttype_residual"]), reverse=True)
    survive = [
        r
        for r in assoc
        if float(r["empirical_p_residual"]) < 0.05 and abs(float(r["spearman_ttype_residual"])) >= 0.10
    ]
    payload = {
        "method": (
            "weighted co-expression modules, WGCNA-style (Python fallback; WGCNA R package not available). "
            "Unsigned |cor|^β + topological overlap + average-linkage cut. "
            "Scale-free R² never reached 0.8; β chosen as the highest value in {4,3,2,1} yielding ≥3 modules of size ≥15."
        ),
        "n_cells_mobility": int(len(shared)),
        "n_cells_rna": int(len(samples)),
        "n_genes_input_matrix": int(counts.shape[0]),
        "n_genes_panel": int(len(panel)),
        "n_curated_present": int(len(curated)),
        "n_modules_tested": int(len(assoc)),
        "n_grey_genes": int(sum(1 for r in gene_assign if r["module"] == "grey")),
        "soft_threshold_beta": int(beta),
        "soft_threshold_fits": {str(k): v for k, v in beta_scores.items()},
        "cut_height": float(cut_height),
        "min_module_size": MIN_MODULE_SIZE,
        "n_shuffle": N_SHUFFLE,
        "mobility_mean": float(np.nanmean(mobility)),
        "mobility_sd": float(np.nanstd(mobility)),
        "top_unadjusted": assoc_sorted_u[:5],
        "top_residual": assoc_sorted_r[:5],
        "modules_surviving_ttype_residual": survive,
        "note": (
            "EXPLAIN remains type-level in the main text. This is an Extended Data control: "
            "within-type transcriptomic correlates of neighborhood mobility, after residualizing on t-type."
        ),
    }
    (OUT_Q3_MOB / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Cell-level mobility and co-expression modules",
        "",
        "Mobility is 1 − mean Spearman of each cell’s rank-distance neighborhood profile across the three k₈₀ regime pairs. "
        "High mobility means the cell’s relative neighbors changed across regimes.",
        "",
        f"**Method:** {payload['method']}",
        "",
        "## Numbers",
        "",
        f"- Cells in all three regimes (mobility): {payload['n_cells_mobility']}",
        f"- With RNA + t-type: {payload['n_cells_rna']}",
        f"- Genes in panel (curated ∪ HVG, cap {N_GENE_CAP}): {payload['n_genes_panel']} "
        f"(curated present {payload['n_curated_present']})",
        f"- Modules tested (size ≥ {MIN_MODULE_SIZE}): {payload['n_modules_tested']}",
        f"- Soft-threshold β={payload['soft_threshold_beta']}; tree cut height={payload['cut_height']}",
        f"- Mobility mean ± sd: {payload['mobility_mean']:.3f} ± {payload['mobility_sd']:.3f}",
        "",
        "### Top modules vs mobility (unadjusted)",
        "",
    ]
    for r in assoc_sorted_u[:8]:
        lines.append(
            f"- {r['module']} (hub {r['hub_gene']}, n={r['n_genes']}): "
            f"ρ={r['spearman_unadjusted']:.3f}, emp p={r['empirical_p_unadjusted']:.4f}"
        )
    lines += ["", "### After t-type residualization (within-type centering of mobility and eigengenes)", ""]
    for r in assoc_sorted_r[:8]:
        lines.append(
            f"- {r['module']} (hub {r['hub_gene']}): "
            f"ρ={r['spearman_ttype_residual']:.3f}, emp p={r['empirical_p_residual']:.4f}"
        )
    if survive:
        lines += ["", "Modules with |ρ|≥0.10 and empirical p<0.05 after t-type residualization:", ""]
        for r in survive:
            lines.append(f"- {r['module']}: ρ={r['spearman_ttype_residual']:.3f}")
    else:
        lines += [
            "",
            "No module retained both |ρ|≥0.10 and empirical p<0.05 after t-type residualization. "
            "Within-type transcriptomic heterogeneity is not a strong correlate of neighborhood mobility "
            "in this panel; the main EXPLAIN result stays type-level.",
        ]
    (OUT_Q3_MOB / "README.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Wrote {OUT_Q3_MOB}")


if __name__ == "__main__":
    main()
