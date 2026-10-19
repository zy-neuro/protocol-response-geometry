#!/usr/bin/env python3
"""WGCNA (R package) modules vs mobility.

Reuses results/q3_mobility/mobility_scores.csv (does not redefine
mobility). Gene panel, join, log-CPM, and t-type residualization match the
Python fallback; only module construction is WGCNA (Langfelder & Horvath).

Writes wgcna_* tables under results/q3_mobility/. Leaves the Python
fallback CSVs in place for audit.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q3_mobility_modules as q3  # noqa: E402
from paths import ORIG_JOIN, OUT_Q3_MOB, REPO, REV  # noqa: E402

MIN_MODULE_SIZE = 20
R_LIB = REV / ".r-lib"
R_SCRIPT = REV / "scripts/q3_wgcna_modules.R"
WORK = OUT_Q3_MOB / "_wgcna_work"


def write_expr_csv(path: Path, sample_ids: list[str], genes: list[str], expr: np.ndarray) -> None:
    """expr: genes × cells. File is cells × genes with sample_id in column 0."""
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(expr, dtype=np.float64).T
    with open(path, "w", newline="") as f:
        f.write("sample_id," + ",".join(genes) + "\n")
        for sid, row in zip(sample_ids, x):
            f.write(sid + "," + ",".join(f"{v:.6g}" for v in row) + "\n")


def load_panel_from_mobility() -> tuple[list[str], np.ndarray, list[str], np.ndarray, list[str], list[str], dict]:
    scores = q3.load_csv(OUT_Q3_MOB / "mobility_scores.csv")
    n_mob = len(scores)
    if n_mob != 4049:
        print(f"WARNING: mobility_scores.csv has n={n_mob} (expected 4049)", flush=True)

    join_rows = q3.load_csv(ORIG_JOIN)
    join = {r["file_stem"]: r for r in join_rows if r.get("file_stem")}

    keep_samples: list[str] = []
    stem_by_sample: dict[str, str] = {}
    ttype_by_sample: dict[str, str] = {}
    mob_by_sample: dict[str, float] = {}
    for rec in scores:
        sid = (rec.get("transcriptomics_sample_id") or "").strip()
        tt = (rec.get("t_type") or "").strip()
        if not sid:
            j = join.get(rec.get("file_stem") or "", {})
            sid = (j.get("transcriptomics_sample_id") or "").strip()
            tt = tt or (j.get("t_type") or "").strip()
        if not sid or not tt:
            continue
        keep_samples.append(sid)
        stem_by_sample[sid] = rec["file_stem"]
        ttype_by_sample[sid] = tt
        mob_by_sample[sid] = float(rec["mobility"])
    keep_samples = sorted(set(keep_samples))
    print(f"Mobility cells: {n_mob}; with RNA+t-type ids: {len(keep_samples)}", flush=True)

    genes, counts, samples = q3.load_counts(q3.nemo_counts_csv(), keep_samples)
    print(f"Loaded counts {counts.shape} (genes x cells)", flush=True)
    expr_all = q3.log_cpm(counts)

    curated_wanted = q3.read_gene_list(q3.CURATED)
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
        if len(panel) >= q3.N_GENE_CAP:
            break
    idx = np.array([gset[g] for g in panel], dtype=int)
    expr = expr_all[idx]
    keep_g = expr.var(axis=1) > 0
    panel = [g for g, ok in zip(panel, keep_g) if ok]
    expr = expr[keep_g]
    print(f"Module gene panel: {len(panel)} (curated present {len(curated)})", flush=True)

    labels = [ttype_by_sample[s] for s in samples]
    mob = np.array([mob_by_sample[s] for s in samples], dtype=float)
    stems = [stem_by_sample[s] for s in samples]
    info = {
        "n_cells_mobility": int(n_mob),
        "n_cells_rna": int(len(samples)),
        "n_genes_input_matrix": int(counts.shape[0]),
        "n_genes_panel": int(len(panel)),
        "n_curated_present": int(len(curated)),
        "mobility_mean": float(np.nanmean([float(r["mobility"]) for r in scores])),
        "mobility_sd": float(np.nanstd([float(r["mobility"]) for r in scores])),
        "curated": curated,
    }
    return samples, expr, panel, mob, stems, labels, info


def run_wgcna(expr_csv: Path) -> None:
    if not (R_LIB / "WGCNA").exists() and not any(R_LIB.glob("WGCNA")):
        # package folder may be WGCNA
        pass
    env = os.environ.copy()
    env["R_LIBS_USER"] = str(R_LIB)
    env["R_LIBS"] = str(R_LIB)
    cmd = [
        "Rscript",
        str(R_SCRIPT),
        str(expr_csv),
        str(OUT_Q3_MOB),
        str(MIN_MODULE_SIZE),
        str(q3.SEED),
    ]
    print("Running", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise SystemExit(
            f"WGCNA R script failed with exit {proc.returncode}. "
            "Not falling back to Python modules."
        )


def associate(samples, stems, labels, mob, info) -> None:
    me_path = OUT_Q3_MOB / "wgcna_module_eigengenes.csv"
    genes_path = OUT_Q3_MOB / "wgcna_module_genes.csv"
    table_path = OUT_Q3_MOB / "wgcna_module_table.csv"
    fit_path = OUT_Q3_MOB / "wgcna_construction.csv"
    me_rows = q3.load_csv(me_path)
    gene_rows = q3.load_csv(genes_path)
    mod_rows = q3.load_csv(table_path)
    fit = q3.load_csv(fit_path)[0]

    me_cols = [c for c in me_rows[0].keys() if c != "transcriptomics_sample_id"]
    # Drop grey from association
    me_cols_tested = [c for c in me_cols if c.replace("ME", "").lower() != "grey"]

    sid_order = [r["transcriptomics_sample_id"] for r in me_rows]
    pos = {s: i for i, s in enumerate(samples)}
    idx = np.array([pos[s] for s in sid_order], dtype=int)
    mob_u = np.asarray(mob, dtype=float)[idx]
    groups = np.asarray(labels, dtype=object)[idx]
    stems_u = [stems[i] for i in idx]

    curated_set = set(info["curated"])
    n_curated_by_mod = Counter(
        r["module"] for r in gene_rows if r["gene"] in curated_set
    )

    for r in mod_rows:
        color = r["module"]
        r["n_curated"] = int(n_curated_by_mod.get(color, 0))
        members = [g["gene"] for g in gene_rows if g["module"] == color]
        r["genes"] = ";".join(members)
    q3.write_csv(table_path, mod_rows)

    ME = np.column_stack(
        [[float(r[c]) for r in me_rows] for c in me_cols_tested]
    ) if me_cols_tested else np.zeros((len(me_rows), 0))

    type_n = Counter(groups.tolist())
    keep_cell = np.array([type_n[t] >= 3 for t in groups], dtype=bool)
    mob_res = q3.residualize_by_group(mob_u, groups)

    color_of_me = {c: c[2:] if c.startswith("ME") else c for c in me_cols_tested}
    table_by_color = {r["module"]: r for r in mod_rows}

    assoc = []
    rng_u = np.random.default_rng(q3.SEED)
    rng_r = np.random.default_rng(q3.SEED + 1)
    for j, col in enumerate(me_cols_tested):
        color = color_of_me[col]
        rec_mod = table_by_color.get(color, {})
        me = ME[:, j]
        me_res = q3.residualize_by_group(me, groups)
        rho = q3.spearman(me, mob_u)
        p_an = q3.spearman_p(rho, int(np.isfinite(me).sum()))
        p_emp = q3.empirical_spearman_p(me, mob_u, rng_u, q3.N_SHUFFLE)
        rho_r = q3.spearman(me_res[keep_cell], mob_res[keep_cell])
        n_res = int(np.isfinite(me_res[keep_cell]).sum())
        p_an_r = q3.spearman_p(rho_r, n_res)
        p_emp_r = q3.empirical_spearman_p(me_res[keep_cell], mob_res[keep_cell], rng_r, q3.N_SHUFFLE)
        assoc.append(
            {
                "module": color,
                "me_column": col,
                "n_genes": int(float(rec_mod.get("n_genes") or 0)),
                "n_curated": int(float(rec_mod.get("n_curated") or 0)),
                "hub_gene": rec_mod.get("hub_gene") or "",
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
            fdrs = q3.bh_fdr([float(r[key_p]) for r in assoc])
            for r, f in zip(assoc, fdrs):
                r[key_f] = f
    q3.write_csv(OUT_Q3_MOB / "wgcna_module_mobility_association.csv", assoc)

    me_out = []
    me_by_sid = {r["transcriptomics_sample_id"]: r for r in me_rows}
    for i, sid in enumerate(sid_order):
        rec = {
            "file_stem": stems_u[i],
            "transcriptomics_sample_id": sid,
            "t_type": str(groups[i]),
            "mobility": float(mob_u[i]),
            "mobility_ttype_residual": float(mob_res[i]),
        }
        src = me_by_sid[sid]
        for c in me_cols:
            rec[c] = src[c]
        me_out.append(rec)
    q3.write_csv(me_path, me_out)

    assoc_sorted_u = sorted(assoc, key=lambda r: abs(r["spearman_unadjusted"]), reverse=True)
    assoc_sorted_r = sorted(assoc, key=lambda r: abs(r["spearman_ttype_residual"]), reverse=True)
    survive = [
        r
        for r in assoc
        if float(r["empirical_p_residual"]) < 0.05 and abs(float(r["spearman_ttype_residual"])) >= 0.10
    ]
    beta = int(float(fit["soft_threshold_beta"]))
    r2 = float(fit["scale_free_R2"])
    n_mod = int(float(fit["n_modules"]))
    payload = {
        "method": (
            "WGCNA R package (Langfelder & Horvath; unsigned network, "
            "goodSamplesGenes, pickSoftThreshold, blockwiseModules). "
            "Gene panel = curated excitability list ∪ HVG, cap 2000. "
            "t-type residualization: within-type centering of mobility and eigengenes, "
            "Spearman + empirical p (1000 shuffles, seed 0)."
        ),
        "network_type": "unsigned",
        "wgcna_version": fit.get("wgcna_version"),
        "n_cells_mobility": info["n_cells_mobility"],
        "n_cells_rna": info["n_cells_rna"],
        "n_cells_used": int(float(fit["n_cells_used"])),
        "n_genes_input_matrix": info["n_genes_input_matrix"],
        "n_genes_panel": info["n_genes_panel"],
        "n_genes_used": int(float(fit["n_genes_used"])),
        "n_curated_present": info["n_curated_present"],
        "n_modules_tested": int(len(assoc)),
        "n_grey_genes": int(float(fit["n_grey_genes"])),
        "soft_threshold_beta": beta,
        "scale_free_R2": r2,
        "beta_selection": fit.get("beta_selection"),
        "wgcna_power_estimate": fit.get("wgcna_power_estimate"),
        "min_module_size": MIN_MODULE_SIZE,
        "merge_cut_height": float(fit.get("merge_cut_height") or 0.25),
        "n_shuffle": q3.N_SHUFFLE,
        "mobility_mean": info["mobility_mean"],
        "mobility_sd": info["mobility_sd"],
        "top_unadjusted": assoc_sorted_u[:5],
        "top_residual": assoc_sorted_r[:5],
        "modules_surviving_ttype_residual": survive,
        "max_abs_rho_unadjusted": (
            float(max(abs(r["spearman_unadjusted"]) for r in assoc)) if assoc else float("nan")
        ),
        "max_abs_rho_residual": (
            float(max(abs(r["spearman_ttype_residual"]) for r in assoc)) if assoc else float("nan")
        ),
        "python_fallback_kept": [
            "module_table.csv",
            "module_genes.csv",
            "module_eigengenes.csv",
            "module_mobility_association.csv",
            "summary.json",
        ],
        "note": (
            "EXPLAIN remains type-level in the main text. This is an Extended Data control. "
            "Python WGCNA-style fallback tables are retained for audit; WGCNA files are the reported result."
        ),
    }
    (OUT_Q3_MOB / "wgcna_summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Cell-level mobility and co-expression modules",
        "",
        "Mobility is 1 − mean Spearman of each cell’s rank-distance neighborhood profile across the three k₈₀ regime pairs. "
        "High mobility means the cell’s relative neighbors changed across regimes.",
        "",
        f"**Reported result:** {payload['method']}",
        "",
        f"- Network: unsigned; minModuleSize = {MIN_MODULE_SIZE}; mergeCutHeight = 0.25; seed = {q3.SEED}.",
        f"- Soft-threshold β = {beta}; scale-free R² = {r2:.3f} ({fit.get('beta_selection')}).",
        f"- Modules tested (non-grey): {n_mod}. Grey/unassigned genes: {int(float(fit['n_grey_genes']))}.",
        "",
        "Python WGCNA-style fallback tables (`module_*.csv`, `summary.json`) are kept for audit. "
        "Use the `wgcna_*` files as the reported result.",
        "",
        "## Numbers",
        "",
        f"- Cells in all three regimes (mobility): {payload['n_cells_mobility']}",
        f"- With RNA + t-type: {payload['n_cells_rna']} (WGCNA used {payload['n_cells_used']})",
        f"- Genes in panel (curated ∪ HVG, cap {q3.N_GENE_CAP}): {payload['n_genes_panel']} "
        f"(curated present {payload['n_curated_present']}; after goodSamplesGenes {payload['n_genes_used']})",
        f"- Modules tested (size ≥ {MIN_MODULE_SIZE}, grey dropped): {payload['n_modules_tested']}",
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
        lines += [
            "",
            "Modules with |ρ|≥0.10 and empirical p<0.05 after t-type residualization:",
            "",
        ]
        for r in survive:
            lines.append(
                f"- {r['module']}: ρ={r['spearman_ttype_residual']:.3f}, emp p={r['empirical_p_residual']:.4f}"
            )
    else:
        lines += [
            "",
            "No module retained both |ρ|≥0.10 and empirical p<0.05 after t-type residualization. "
            "Within-type transcriptomic heterogeneity is not a strong correlate of neighborhood mobility "
            "in this panel; the main EXPLAIN result stays type-level.",
        ]
    lines += [
        "",
        "## Files",
        "",
        "Reported result (WGCNA R package):",
        "",
        "- `wgcna_soft_threshold.csv` — pickSoftThreshold fit (powers 1–20)",
        "- `wgcna_module_table.csv` — modules (grey flagged; not tested)",
        "- `wgcna_module_genes.csv` — gene membership + kME",
        "- `wgcna_module_eigengenes.csv` — ME matrix + mobility residuals",
        "- `wgcna_module_mobility_association.csv` — unadjusted and t-type-residual Spearman",
        "- `wgcna_summary.json` — β, R², n modules, n cells, n genes",
        "- `wgcna_construction.csv`, `wgcna_gene_panel.csv` — run metadata",
        "",
        "Audit (Python WGCNA-style fallback; not the reported result): `module_*.csv`, `summary.json`.",
        "Mobility scores (unchanged): `mobility_scores.csv`.",
    ]
    (OUT_Q3_MOB / "README.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    OUT_Q3_MOB.mkdir(parents=True, exist_ok=True)
    scores_path = OUT_Q3_MOB / "mobility_scores.csv"
    if not scores_path.exists():
        raise SystemExit(f"Missing {scores_path}; run q3_mobility_modules.py first to score mobility.")

    samples, expr, panel, mob, stems, labels, info = load_panel_from_mobility()
    WORK.mkdir(parents=True, exist_ok=True)
    expr_csv = WORK / "expr.csv"
    skip_wgcna = (
        "--associate-only" in sys.argv
        and (OUT_Q3_MOB / "wgcna_module_eigengenes.csv").exists()
        and (OUT_Q3_MOB / "wgcna_construction.csv").exists()
    )
    if not skip_wgcna:
        print(f"Writing {expr_csv} …", flush=True)
        write_expr_csv(expr_csv, samples, panel, expr)
        q3.write_csv(
            OUT_Q3_MOB / "wgcna_gene_panel.csv",
            [{"gene": g, "curated": int(g in set(info["curated"]))} for g in panel],
        )
        run_wgcna(expr_csv)
    else:
        print("Reusing existing WGCNA construction tables (--associate-only)", flush=True)
    associate(samples, stems, labels, mob, info)
    print(f"Wrote WGCNA outputs under {OUT_Q3_MOB}")


if __name__ == "__main__":
    main()
