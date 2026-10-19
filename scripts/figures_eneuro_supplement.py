#!/usr/bin/env python3
"""eNeuro supplement figures + tables → figures/supplement/, tables/

Lock: docs/SUPPLEMENT.md
Image ED: 1-1, 1-2, 1-3, 2-1, 3-2, 3-3, 4-2, 4-4. Word ED 3-1 / 4-1 / 4-3 are not written here.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-prg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "figures/supplement"
TAB = REPO / "tables"
PHENO = REPO / "results/phenotype"
JOIN = REPO / "results/m0/m0_molecular_join.csv"
Q1 = REPO / "results/q1/q1_cross_protocol.csv"
Q1_SCALE = REPO / "results/q1/q1_knn_scale.csv"
Q1_REL = REPO / "results/q1_reliability"
Q3_MOB = REPO / "results/q3_mobility"
Q2 = REPO / "results/q2/q2_anchor.csv"
Q2W = REPO / "results/q2/q2_within_cre.csv"
PAIRS = REPO / "results/q3a/q3a_pair_tags.csv"
CONC = REPO / "results/q3a/q3a_S_concordance.csv"
CONC_1000 = REPO / "results/q3a_robustness/concordance_null_1000.csv"
Q3B_PRIMARY_1000 = REPO / "results/q3b/q3b_primary_null_1000.csv"
SUBC = REPO / "results/q3a/q3a_subclass_concordance.csv"
TYPES = REPO / "results/q3a/q3a_type_counts.csv"
S_FILES = {
    "long_pulse": REPO / "results/q3a/q3a_S_long_pulse.csv",
    "ramp": REPO / "results/q3a/q3a_S_ramp.csv",
    "brief_pulse": REPO / "results/q3a/q3a_S_brief_pulse.csv",
}
Q3B = REPO / "results/q3b/q3b_panel_summary.csv"
MODS = REPO / "results/q3c/q3c_module_tests.csv"
PHENO_SUM = REPO / "results/phenotype/phenotype_summary.json"
LOAD_DIR = REPO / "results/phenotype"
VAR_FILES = {
    "long_pulse": PHENO / "long_pulse_pca_variance.csv",
    "ramp": PHENO / "ramp_pca_variance.csv",
    "brief_pulse": PHENO / "brief_pulse_pca_variance.csv",
}

LONG, RAMP, BRIEF = "#3B8FB0", "#C4843C", "#2A9A74"
INK, MUTED = "#3D4A55", "#9AA3AD"
PURPLE, ROSE = "#7A6B8A", "#B88484"
PCOL = {"long_pulse": LONG, "ramp": RAMP, "brief_pulse": BRIEF}
SHORT = {"long_pulse": "Long", "ramp": "Ramp", "brief_pulse": "Brief"}
PROTOCOLS = ["long_pulse", "ramp", "brief_pulse"]
TAG = {"stable": "#6E8B9A", "selective": "#B88484", "intermediate": "#C5B6A4"}
SAND, HIER, REF = "#B8A07A", "#6A8B84", "#5A6570"
LAB_T, LAB_CRE, LAB_HIER = "#6B7280", "#8B6B4A", "#4A4A4A"
LAB_T_CRE = mcolors.to_hex(
    tuple((a + b) / 2 for a, b in zip(mcolors.to_rgb(LAB_T), mcolors.to_rgb(LAB_CRE)))
)
STABLE_RANK_CV = 0.25
SELECTIVE_RANK_RANGE = 0.55


def soft_bar(hex_c, a=0.48):
    r, g, b = mcolors.to_rgb(hex_c)
    return (1 - a + a * r, 1 - a + a * g, 1 - a + a * b)


def soft_cmap(hex_c, name="soft"):
    c = mcolors.hex2color(hex_c)
    return mcolors.LinearSegmentedColormap.from_list(
        name, [(1, 1, 1), tuple(0.72 + 0.28 * x for x in c), tuple(0.35 + 0.65 * x for x in c), c]
    )


def style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "axes.edgecolor": MUTED,
            "xtick.color": "#5A6570",
            "ytick.color": "#5A6570",
            "axes.labelcolor": INK,
            "text.color": INK,
            "hatch.linewidth": 0.7,
            "pdf.fonttype": 42,
        }
    )


def letter(ax, s):
    # Offset in points (not axes fraction) so full-width panels share the same figure-x as half-width ones.
    ax.annotate(
        s,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-14, 4),
        textcoords="offset points",
        fontweight="bold",
        fontsize=12,
        va="bottom",
        ha="left",
        clip_on=False,
    )


def save_fig(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.png", dpi=300, facecolor="white", pad_inches=0.12)
    fig.savefig(FIG / f"{stem}.pdf", facecolor="white", pad_inches=0.12)
    plt.close(fig)
    print("wrote", FIG / stem)


def ylim_from_zero(ax, hi=None, pad=0.08):
    """Pin the y origin to 0; pad only above the data."""
    if hi is None:
        hi = ax.get_ylim()[1]
    hi = max(float(hi), 0.0)
    ax.set_ylim(0.0, hi * (1.0 + pad) if hi > 0 else 1.0)


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    TAB.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


def load_var(path):
    rows = load_csv(path)
    return np.array([float(r["explained_variance_ratio"]) for r in rows], dtype=float)


def load_S_matrix(path):
    rows = load_csv(path)
    types = sorted(set(r["t_type_a"] for r in rows) | set(r["t_type_b"] for r in rows))
    idx = {t: i for i, t in enumerate(types)}
    n = len(types)
    M = np.zeros((n, n))
    for r in rows:
        i, j = idx[r["t_type_a"]], idx[r["t_type_b"]]
        M[i, j] = M[j, i] = float(r["S"])
    return types, M


def retag(rank_cv, rank_range, cv_thr, range_thr):
    if rank_cv < cv_thr:
        return "stable"
    if rank_range >= range_thr:
        return "selective"
    return "intermediate"


def short_cre(g):
    g = g.split(";")[0]
    return g.replace("/wt", "").replace("-IRES-Cre", "").replace("-Cre", "")[:22]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_s1():
    """Phenotype construction: feature n, scree, k80."""
    summary = json.loads(PHENO_SUM.read_text())
    fig = plt.figure(figsize=(12.0, 4.0))
    gs = gridspec.GridSpec(1, 3, wspace=0.34, left=0.07, right=0.98, top=0.88, bottom=0.18)

    # A — n features / n cells / k80
    ax = fig.add_subplot(gs[0, 0])
    xs = np.arange(3)
    ns = [summary[p]["n_features"] for p in PROTOCOLS]
    ks = [summary[p]["k80"] for p in PROTOCOLS]
    cells = [summary[p]["n_cells"] for p in PROTOCOLS]
    ax.bar(xs - 0.18, ns, 0.35, color=[soft_bar(c, 0.38) for c in (LONG, RAMP, BRIEF)], edgecolor=[LONG, RAMP, BRIEF], lw=0.9)
    ax.bar(xs + 0.18, ks, 0.35, color="white", edgecolor=[LONG, RAMP, BRIEF], lw=1.1)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{SHORT[p]}\nn={c}" for p, c in zip(PROTOCOLS, cells)],
        fontsize=8,
    )
    for i, (n, k) in enumerate(zip(ns, ks)):
        ax.text(i - 0.18, n + 0.6, str(n), ha="center", fontsize=7.5)
        ax.text(i + 0.18, k + 0.6, str(k), ha="center", fontsize=7.5)
    ax.set_ylabel("Count")
    ylim_from_zero(ax, max(ns), pad=0.25)
    ax.legend(
        handles=[
            Patch(facecolor=soft_bar(LONG, 0.38), edgecolor=LONG, label="features"),
            Patch(facecolor="white", edgecolor=INK, lw=1.1, label="k₈₀"),
        ],
        frameon=False, fontsize=7, loc="upper right",
    )
    letter(ax, "A")

    # B — cumulative variance
    ax = fig.add_subplot(gs[0, 1])
    for p in PROTOCOLS:
        ve = load_var(VAR_FILES[p])
        ax.plot(range(1, len(ve) + 1), np.cumsum(ve), color=PCOL[p], lw=1.8, label=SHORT[p])
        k = summary[p]["k80"]
        ax.scatter([k], [np.cumsum(ve)[k - 1]], color=PCOL[p], s=36, zorder=3)
    ax.axhline(0.8, ls="--", color=REF, lw=1.0)
    ax.text(0.98, 0.80, "80%", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7, color=MUTED)
    ax.set_xlabel("PC")
    ax.set_ylabel("Cumulative variance")
    ax.set_xlim(1, 25)
    ylim_from_zero(ax, 1.0, pad=0.02)
    ax.legend(frameon=False, fontsize=7.5)
    letter(ax, "B")

    # C — per-PC variance (Long example + note)
    ax = fig.add_subplot(gs[0, 2])
    for p in PROTOCOLS:
        ve = load_var(VAR_FILES[p])[:15]
        ax.plot(range(1, len(ve) + 1), ve, color=PCOL[p], lw=1.4, marker="o", ms=3, label=SHORT[p])
    ax.set_xlabel("PC")
    ax.set_ylabel("Variance fraction")
    ax.legend(frameon=False, fontsize=7.5)
    letter(ax, "C")
    save_fig(fig, "Figure 1-1")


PAIR = "#7A858C"
PAIR_FILL = 0.38
PAIR_LR, PAIR_LB, PAIR_RB = "#1F3F58", "#0E4A47", "#542616"
PAIR_COL = {
    "long_pulse__ramp": PAIR_LR,
    "long_pulse__brief_pulse": PAIR_LB,
    "ramp__brief_pulse": PAIR_RB,
}
PAIR_MK = {
    "long_pulse__ramp": "o",
    "long_pulse__brief_pulse": "s",
    "ramp__brief_pulse": "^",
}


def fig_s2():
    """Q1 robustness: k80 vs k5 D, kNN scale sweep, PC1 loadings."""
    rows = load_csv(Q1)
    scale = load_csv(Q1_SCALE)
    order = ["long_pulse__ramp", "long_pulse__brief_pulse", "ramp__brief_pulse"]
    labs = ["Long–Ramp", "Long–Brief", "Ramp–Brief"]
    fig = plt.figure(figsize=(12.4, 8.6))
    gs = gridspec.GridSpec(2, 3, hspace=0.50, wspace=0.50, left=0.14, right=0.98, top=0.94, bottom=0.08)

    def subset(rule):
        return sorted([r for r in rows if r["k_rule"] == rule], key=lambda r: order.index(r["pair"]))

    k80, k5 = subset("k80"), subset("k5")
    xs = np.arange(3)
    pair_edges = [PAIR_COL[k] for k in order]

    ax = fig.add_subplot(gs[0, 0])
    ax.bar(
        xs - 0.18, [float(r["spearman_D"]) for r in k80], 0.35,
        color="white", edgecolor=pair_edges, lw=1.1,
    )
    ax.bar(
        xs + 0.18, [float(r["spearman_D"]) for r in k5], 0.35,
        color=[soft_bar(c, PAIR_FILL) for c in pair_edges], edgecolor=pair_edges, lw=0.9,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labs, fontsize=7.5)
    ax.set_ylabel("Spearman of cell–cell D")
    ylim_from_zero(ax, 0.50, pad=0.10)
    ax.legend(
        handles=[
            Patch(facecolor="white", edgecolor=PAIR, lw=1.1, label="k₈₀"),
            Patch(facecolor=soft_bar(PAIR, PAIR_FILL), edgecolor=PAIR, lw=0.9, label="5-PC"),
        ],
        frameon=False, fontsize=7, loc="upper right",
    )
    letter(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    chance_by_k = {}
    for r in scale:
        chance_by_k.setdefault(int(float(r["k"])), []).append(float(r["chance_jaccard_mean"]))
    ck = sorted(chance_by_k)
    ax.fill_between(
        ck,
        [min(chance_by_k[k]) for k in ck],
        [max(chance_by_k[k]) for k in ck],
        color=PAIR, alpha=0.12, lw=0, zorder=1,
    )
    ax.plot(ck, [float(np.mean(chance_by_k[k])) for k in ck], color=REF, lw=1.2, ls=":", zorder=2)
    for key, lab in zip(order, labs):
        sub = sorted([r for r in scale if r["pair"] == key], key=lambda r: int(float(r["k"])))
        kk = [int(float(r["k"])) for r in sub]
        ax.plot(
            kk, [float(r["observed_jaccard_mean"]) for r in sub],
            color=PAIR_COL[key], lw=1.6, marker=PAIR_MK[key], ms=5.5,
            markerfacecolor=PAIR_COL[key], markeredgecolor="white", markeredgewidth=0.5,
            zorder=3,
        )
    ax.set_xlabel("Neighborhood size k")
    ax.set_ylabel("Mean kNN Jaccard")
    ymax = max(float(r["observed_jaccard_mean"]) for r in scale)
    ylim_from_zero(ax, ymax)
    ax.set_xlim(0, 640)
    ax.legend(
        handles=[
            Line2D([0], [0], color=PAIR_LR, marker="o", ms=5.5, markeredgecolor="white", lw=1.6, label="Long–Ramp"),
            Line2D([0], [0], color=PAIR_LB, marker="s", ms=5.5, markeredgecolor="white", lw=1.6, label="Long–Brief"),
            Line2D([0], [0], color=PAIR_RB, marker="^", ms=5.5, markeredgecolor="white", lw=1.6, label="Ramp–Brief"),
            Line2D([0], [0], color=REF, ls=":", lw=1.2, label="Chance"),
        ],
        frameon=False, fontsize=6.5, loc="upper left",
    )
    letter(ax, "B")

    ax = fig.add_subplot(gs[0, 2])
    for key, lab in zip(order, labs):
        sub = sorted([r for r in scale if r["pair"] == key], key=lambda r: int(float(r["k"])))
        kk = [int(float(r["k"])) for r in sub]
        ax.plot(
            kk, [float(r["adjusted_jaccard"]) for r in sub],
            color=PAIR_COL[key], lw=1.6, marker=PAIR_MK[key], ms=5.5,
            markerfacecolor=PAIR_COL[key], markeredgecolor="white", markeredgewidth=0.5,
            label=lab,
        )
    ax.set_xlabel("Neighborhood size k")
    ax.set_ylabel("Chance-adjusted Jaccard")
    ymax = max(float(r["adjusted_jaccard"]) for r in scale)
    ylim_from_zero(ax, ymax)
    ax.set_xlim(0, 640)
    ax.legend(
        handles=[
            Line2D([0], [0], color=PAIR_LR, marker="o", ms=5.5, markeredgecolor="white", lw=1.6, label="Long–Ramp"),
            Line2D([0], [0], color=PAIR_LB, marker="s", ms=5.5, markeredgecolor="white", lw=1.6, label="Long–Brief"),
            Line2D([0], [0], color=PAIR_RB, marker="^", ms=5.5, markeredgecolor="white", lw=1.6, label="Ramp–Brief"),
        ],
        frameon=False, fontsize=6.5, loc="upper left",
    )
    letter(ax, "C")

    letters = ["D", "E", "F"]
    for i, p in enumerate(PROTOCOLS):
        ax = fig.add_subplot(gs[1, i])
        raw = load_csv(LOAD_DIR / f"{p}_pca_loadings.csv")
        load_rows, seen = [], set()
        for r in raw:
            key = round(float(r["PC1"]), 6)
            if key in seen:
                continue
            seen.add(key)
            load_rows.append(r)
            if len(load_rows) >= 8:
                break
        names = [r["display"].replace("mean spike 0.", "")[:24] for r in load_rows][::-1]
        vals = [float(r["PC1"]) for r in load_rows][::-1]
        colors = [soft_bar(PCOL[p]) for _ in vals]
        ax.barh(np.arange(len(vals)), vals, color=colors, edgecolor=PCOL[p], lw=0.6, height=0.72)
        ax.set_yticks(np.arange(len(vals)))
        ax.set_yticklabels(names, fontsize=7.5)
        ax.axvline(0, color="#5A6570", lw=0.8)
        ax.set_xlabel("PC1 loading")
        ax.set_title(SHORT[p], fontsize=10, fontweight="bold", color=PCOL[p], pad=3)
        letter(ax, letters[i])
    save_fig(fig, "Figure 1-2")


def fig_s13():
    """Split-half within-regime vs cross-regime 15-NN Jaccard."""
    summary_path = Q1_REL / "split_half_summary.csv"
    cells_path = Q1_REL / "split_half_jaccard_cells.csv"
    if not summary_path.exists() or not cells_path.exists():
        print("skip Figure 1-3 (missing q1_reliability tables)")
        return
    summary = load_csv(summary_path)
    cells = load_csv(cells_path)
    within = np.array([float(r["jaccard_within_k15"]) for r in cells], dtype=float)
    pair_keys = [
        ("long_pulse__ramp", "Long–Ramp"),
        ("long_pulse__brief_pulse", "Long–Brief"),
        ("ramp__brief_pulse", "Ramp–Brief"),
    ]

    def discrete_counts(vals):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if not vals.size:
            return np.array([]), np.array([])
        return np.unique(np.round(vals, 8), return_counts=True)

    pair_vals = []
    for key, lab in pair_keys:
        vals = np.array(
            [float(r[f"jaccard_{key}_samepool"]) for r in cells if r.get(f"jaccard_{key}_samepool") not in ("", None)],
            dtype=float,
        )
        pair_vals.append((key, lab, vals))

    fig = plt.figure(figsize=(11.2, 7.2))
    gs = gridspec.GridSpec(
        2, 2, hspace=0.38, wspace=0.32,
        left=0.09, right=0.97, top=0.94, bottom=0.08,
    )
    slots = [(0, 0), (0, 1), (1, 0)]

    u_w, c_w = discrete_counts(within)
    y_hi = float(c_w.max()) if c_w.size else 1.0
    for _, _, vals in pair_vals:
        _, c_p = discrete_counts(vals)
        if c_p.size:
            y_hi = max(y_hi, float(c_p.max()))
    x_hi = 0.38
    if u_w.size:
        x_hi = max(x_hi, float(np.percentile(within, 99)))

    for i, ((key, lab, vals), (r, c)) in enumerate(zip(pair_vals, slots)):
        ax = fig.add_subplot(gs[r, c])
        col = PAIR_COL[key]
        u_p, c_p = discrete_counts(vals)
        if u_w.size:
            ax.bar(
                u_w, c_w, width=0.020, align="center",
                color=soft_bar(INK, 0.50), edgecolor=INK, lw=0.8, zorder=2,
                label="Within long-square",
            )
        if u_p.size:
            ax.bar(
                u_p, c_p, width=0.020, align="center",
                color=soft_bar(col, PAIR_FILL), edgecolor=col, lw=0.9, zorder=3, label=lab,
            )
        ax.set_xlim(-0.03, x_hi)
        ylim_from_zero(ax, y_hi, pad=0.08)
        ax.set_xlabel("15-NN Jaccard")
        ax.set_title(lab, fontsize=10, fontweight="bold", color=col, pad=3)
        ax.set_ylabel("Cells")
        if i == 0:
            letter(ax, "A")
        ax.legend(frameon=False, fontsize=6.5, loc="upper right")

    ax = fig.add_subplot(gs[1, 1])
    rows_plot = []
    within_row = next((r for r in summary if r["contrast"] == "within_long_square_split_half"), None)
    if within_row:
        rows_plot.append(("Within\nlong-square", float(within_row["observed_jaccard_mean"]), float(within_row["chance_jaccard_mean"]), INK))
    for key, lab in pair_keys:
        r = next((x for x in summary if x["contrast"] == key and x["pool"] == "same_split_half_cells"), None)
        if r:
            rows_plot.append((lab.replace("–", "–\n"), float(r["observed_jaccard_mean"]), float(r["chance_jaccard_mean"]), PAIR_COL[key]))
    xs = np.arange(len(rows_plot))
    obs = [t[1] for t in rows_plot]
    cols = [t[3] for t in rows_plot]
    fills = [soft_bar(c, 0.50 if i == 0 else PAIR_FILL) for i, c in enumerate(cols)]
    ax.bar(xs, obs, 0.55, color=fills, edgecolor=cols, lw=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels([t[0] for t in rows_plot], fontsize=7.5)
    ax.set_ylabel("Mean 15-NN Jaccard")
    ylim_from_zero(ax, max(obs) if obs else 0.12, pad=0.12)
    letter(ax, "B")
    save_fig(fig, "Figure 1-3")


def fig_mobility():
    """Cell-level mobility and module–mobility associations (WGCNA R package)."""
    scores_path = Q3_MOB / "mobility_scores.csv"
    assoc_path = Q3_MOB / "wgcna_module_mobility_association.csv"
    if not scores_path.exists() or not assoc_path.exists():
        print("skip Figure 4-4 (missing q3_mobility WGCNA tables)")
        return
    scores = load_csv(scores_path)
    assoc = load_csv(assoc_path)
    mob = np.array([float(r["mobility"]) for r in scores], dtype=float)
    fig_h = max(3.9, 0.28 * len(assoc) + 1.8)
    fig = plt.figure(figsize=(10.8, fig_h))
    gs = gridspec.GridSpec(1, 2, wspace=0.36, left=0.08, right=0.98, top=0.88, bottom=0.22)

    ax = fig.add_subplot(gs[0, 0])
    ax.hist(mob, bins=40, color=soft_bar(INK, 0.35), edgecolor=INK, lw=0.6)
    ax.axvline(float(np.nanmean(mob)), color=INK, lw=1.1, ls="--")
    ax.set_xlabel("Mobility (1 − mean rank-distance Spearman)")
    ax.set_ylabel("Cells")
    ax.text(0.98, 0.96, f"n={len(mob)}", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK)
    letter(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    assoc = sorted(assoc, key=lambda r: abs(float(r["spearman_unadjusted"])), reverse=True)
    names = [r["module"] for r in assoc]
    rho_u = [float(r["spearman_unadjusted"]) for r in assoc]
    rho_r = [float(r["spearman_ttype_residual"]) for r in assoc]
    ys = np.arange(len(names))
    ax.barh(ys + 0.16, rho_u, 0.30, color=soft_bar(INK, 0.35), edgecolor=INK, lw=0.7, label="Unadjusted")
    ax.barh(ys - 0.16, rho_r, 0.30, color="white", edgecolor=LAB_T, lw=1.0, label="t-type residual")
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#5A6570", lw=0.8)
    ax.set_xlabel("Spearman ρ (module eigengene vs mobility)")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    letter(ax, "B")
    save_fig(fig, "Figure 4-4")


def fig_s3():
    """Q2 controls: A_p transparency + within-Cre side by side (η²/kNN in main Fig 2 only)."""
    q2 = {r["protocol"]: r for r in load_csv(Q2)}
    within = load_csv(Q2W)
    fig = plt.figure(figsize=(12.2, 5.4))
    gs = gridspec.GridSpec(
        1, 2,
        width_ratios=[1.0, 1.2],
        wspace=0.42,
        left=0.09, right=0.96, top=0.90, bottom=0.14,
    )

    # A — A_p (small; show transparently)
    ax = fig.add_subplot(gs[0, 0])
    xs = np.arange(3)
    w = 0.22
    a_t = [float(q2[p]["A_t_type"]) for p in PROTOCOLS]
    a_c = [float(q2[p]["A_cre"]) for p in PROTOCOLS]
    a_h = [float(q2[p]["A_hier"]) for p in PROTOCOLS]
    a_r = [float(q2[p]["A_t_type_after_cre_residual"]) for p in PROTOCOLS]
    ax.bar(xs - 1.5 * w, a_t, w, color=soft_bar(LAB_T, 0.42), edgecolor=LAB_T, lw=0.8, label="A_t-type")
    ax.bar(xs - 0.5 * w, a_c, w, color=soft_bar(LAB_CRE, 0.50), edgecolor=LAB_CRE, lw=0.8, label="A_Cre")
    ax.bar(xs + 0.5 * w, a_h, w, color=soft_bar(LAB_HIER, 0.40), edgecolor=LAB_HIER, lw=0.8, label="A_hier")
    ax.bar(xs + 1.5 * w, a_r, w, color=soft_bar(LAB_T_CRE, 0.55), edgecolor=LAB_T_CRE, lw=0.8, label="A_t|Cre")
    ax.axhline(0, color="#5A6570", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[p] for p in PROTOCOLS])
    ax.set_ylabel("A_p (distance alignment)")
    ylim_from_zero(ax, 0.25, pad=0)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right", ncol=2)
    ax.text(
        0.02, 0.96, "binary A_p is small — shown for transparency",
        transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=MUTED,
    )
    letter(ax, "A")

    # B — within-Cre η² (Long)
    ax = fig.add_subplot(gs[0, 1])
    lp = [r for r in within if r["protocol"] == "long_pulse"]
    lp = sorted(lp, key=lambda r: -int(r["n_cells"]))[:10]
    ys = np.arange(len(lp))
    ax.barh(ys, [float(r["eta2"]) for r in lp], color=soft_bar(LAB_T, 0.42), edgecolor=LAB_T, lw=0.8, height=0.7)
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [f"{short_cre(r['cre_line'])}\n(n={r['n_cells']})" for r in lp],
        fontsize=6.5,
        linespacing=0.95,
    )
    ax.invert_yaxis()
    ax.set_xlabel("η² within Cre (Long)")
    letter(ax, "B")
    save_fig(fig, "Figure 2-1")


def fig_s4():
    """Q3: full S maps + tag threshold sensitivity + subclass."""
    pairs = load_csv(PAIRS)
    fig = plt.figure(figsize=(12.4, 8.0))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.35, 1.0], hspace=0.35, wspace=0.28, left=0.06, right=0.96, top=0.94, bottom=0.08)

    # A–C full S matrices
    for i, p in enumerate(PROTOCOLS):
        ax = fig.add_subplot(gs[0, i])
        types, M = load_S_matrix(S_FILES[p])
        im = ax.imshow(M, cmap=soft_cmap(PCOL[p], f"S{p}"), origin="upper")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(SHORT[p], color=PCOL[p], fontweight="bold", fontsize=11)
        letter(ax, "ABC"[i])
        ax.set_ylabel(f"t-types (n={len(types)})", fontsize=8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Type–type D", fontsize=8)

    # D — threshold sensitivity
    ax = fig.add_subplot(gs[1, 0])
    ranges = [0.45, 0.50, 0.55, 0.60, 0.65]
    sel_frac = []
    for thr in ranges:
        tags = [retag(float(r["rank_cv"]), float(r["rank_range"]), STABLE_RANK_CV, thr) for r in pairs]
        c = Counter(tags)
        sel_frac.append(100 * c["selective"] / len(pairs))
    ax.plot(ranges, sel_frac, color=INK, marker="o", lw=1.8)
    ax.axvline(SELECTIVE_RANK_RANGE, ls="--", color=REF, lw=1.0)
    ax.set_xticks(ranges)
    ax.set_xticklabels([f"{t:.2f}" for t in ranges])
    ax.set_xlim(0.43, 0.67)
    ax.set_xlabel("Selective rank-range cutoff")
    ax.set_ylabel("% selective pairs")
    letter(ax, "D")

    # E — vary CV at fixed range
    ax = fig.add_subplot(gs[1, 1])
    cvs = [0.15, 0.20, 0.25, 0.30, 0.35]
    stab_frac = []
    for thr in cvs:
        tags = [retag(float(r["rank_cv"]), float(r["rank_range"]), thr, SELECTIVE_RANK_RANGE) for r in pairs]
        c = Counter(tags)
        stab_frac.append(100 * c["stable"] / len(pairs))
    ax.plot(cvs, stab_frac, color=INK, marker="o", lw=1.8)
    ax.axvline(STABLE_RANK_CV, ls="--", color=REF, lw=1.0)
    ax.set_xticks(cvs)
    ax.set_xticklabels([f"{t:.2f}" for t in cvs])
    ax.set_xlim(0.13, 0.37)
    ax.set_xlabel("Stable rank-CV cutoff")
    ax.set_ylabel("% stable pairs")
    letter(ax, "E")

    # F — subclass concordance
    ax = fig.add_subplot(gs[1, 2])
    sub = load_csv(SUBC)
    order = ["long_pulse__ramp", "long_pulse__brief_pulse", "ramp__brief_pulse"]
    sub = sorted(sub, key=lambda r: order.index(r["pair"]))
    labs = ["Long–Ramp", "Long–Brief", "Ramp–Brief"]
    xs = np.arange(3)
    edges = [PAIR_COL[k] for k in order]
    ax.bar(xs, [float(r["spearman_S"]) for r in sub], 0.55, color=[soft_bar(c, PAIR_FILL) for c in edges], edgecolor=edges, lw=0.9)
    for x, r in zip(xs, sub):
        ax.text(x, float(r["spearman_S"]) + 0.02, f"{float(r['spearman_S']):.2f}", ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labs, fontsize=7.5)
    ax.set_ylabel("Spearman of subclass S")
    ylim_from_zero(ax, 1.05, pad=0)
    letter(ax, "F")
    save_fig(fig, "Figure 3-2")


def fig_s5():
    """Q4 expansion: 3 RNA panels + marker-control modules only (excitability modules live in main Fig 4C)."""
    panels = load_csv(Q3B)
    mods = [r for r in load_csv(MODS) if r.get("role") == "marker_control"]
    fig = plt.figure(figsize=(12.0, 4.0))
    gs = gridspec.GridSpec(1, 3, wspace=0.38, left=0.07, right=0.98, top=0.86, bottom=0.22)

    # A — ρ(T, meanS) across gene panels
    ax_a = fig.add_subplot(gs[0, 0])
    order = ["primary", "curated", "hvg500"]
    labs = ["primary\n(500)", "curated\n(263)", "HVG500"]
    pan = {r["panel"]: r for r in panels}
    xs = np.arange(3)
    vals = [float(pan[k]["spearman_T_vs_meanS"]) for k in order]
    ax_a.bar(xs, vals, 0.55, color=soft_bar(PURPLE), edgecolor=PURPLE, lw=0.9)
    for x, v in zip(xs, vals):
        ax_a.text(x, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(labs, fontsize=7.5)
    ax_a.set_ylabel("ρ(T, mean S)")
    ylim_from_zero(ax_a, 0.70, pad=0)
    letter(ax_a, "A")

    # B — ρ(T, rewrite) across panels
    ax_b = fig.add_subplot(gs[0, 1])
    vals_r = [float(pan[k]["spearman_T_vs_rank_range"]) for k in order]
    ax_b.bar(xs, vals_r, 0.55, color=soft_bar(ROSE), edgecolor=ROSE, lw=0.9)
    for x, v in zip(xs, vals_r):
        ax_b.text(x, max(v, 0) + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels(labs, fontsize=7.5)
    ax_b.set_ylabel("ρ(T, rank-range)")
    ylim_from_zero(ax_b, 0.25, pad=0)
    letter(ax_b, "B")

    # C — marker controls only (not the excitability modules already in main Fig 4C)
    ax_c = fig.add_subplot(gs[0, 2])
    w = 0.32
    shared = [float(r["spearman_D_vs_meanS"]) for r in mods]
    rewrite = [float(r["spearman_D_vs_rank_range"]) for r in mods]
    labels = [r["module"].replace("_marker", "") for r in mods]
    ax_c.bar(xs - w / 2, shared, w, color=soft_bar(PURPLE), edgecolor=PURPLE, lw=0.9, label="shared structure")
    ax_c.bar(xs + w / 2, rewrite, w, color=soft_bar(ROSE), edgecolor=ROSE, lw=0.9, label="regime-selective rewrite")
    ax_c.axhline(0, color="#5A6570", lw=0.7)
    ax_c.set_xticks(xs)
    ax_c.set_xticklabels(labels, fontsize=8)
    ax_c.set_ylabel("Spearman of marker-gene distance")
    ymin = min(0, min(shared + rewrite) - 0.08)
    ymax = max(shared + rewrite) + 0.10
    ax_c.set_ylim(ymin, ymax)
    ax_c.legend(frameon=False, fontsize=6.5, loc="upper right")
    letter(ax_c, "C")

    save_fig(fig, "Figure 4-2")


def fig_s32():
    """Figure 3-3 from saved equal-N / split-half summaries (no bootstrap rerun)."""
    equal_path = REPO / "results/q3a_robustness/equalN_summary.csv"
    reliab_path = REPO / "results/q3a_robustness/within_regime_split_half_summary.csv"
    if not equal_path.exists() or not reliab_path.exists():
        print("skip Figure 3-3 (missing q3a_robustness tables)")
        return
    equal = load_csv(equal_path)
    reliab = load_csv(reliab_path)
    pair_labs = ["long_ramp", "long_brief", "ramp_brief"]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.16, wspace=0.34)
    xs = np.arange(3)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(MUTED)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(colors="#5A6570")

    ax = axes[0]
    means = np.array([next(float(r["equalN_mean"]) for r in equal if r["pair"] == lab) for lab in pair_labs])
    lo = np.array([next(float(r["equalN_q025"]) for r in equal if r["pair"] == lab) for lab in pair_labs])
    hi = np.array([next(float(r["equalN_q975"]) for r in equal if r["pair"] == lab) for lab in pair_labs])
    obs_v = np.array([next(float(r["observed_full_n"]) for r in equal if r["pair"] == lab) for lab in pair_labs])
    pair_keys = ["long_pulse__ramp", "long_pulse__brief_pulse", "ramp__brief_pulse"]
    pair_edges = [PAIR_COL[k] for k in pair_keys]
    ax.bar(
        xs, means, 0.55, color=[soft_bar(c, PAIR_FILL) for c in pair_edges], edgecolor=pair_edges, lw=0.9,
        yerr=np.vstack([means - lo, hi - means]), capsize=3,
        error_kw=dict(ecolor="#5A6570", lw=0.9),
        zorder=2,
    )
    ax.scatter(xs, obs_v, color=INK, zorder=4, s=32)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Long–Ramp", "Long–Brief", "Ramp–Brief"], fontsize=8)
    ax.set_ylabel("Spearman of type–type S", color=INK)
    ylim_from_zero(ax, 1.0, pad=0)
    ax.legend(
        handles=[
            Patch(facecolor=soft_bar(PAIR), edgecolor=PAIR, label="Equal-N mean ± 95% boot"),
            plt.Line2D(
                [0], [0], marker="o", color="none",
                markerfacecolor=INK, markersize=6, label="Full-n observed",
            ),
        ],
        frameon=False, fontsize=7, loc="lower center",
        bbox_to_anchor=(0.5, 1.04), ncol=2,
    )
    letter(ax, "A")

    ax = axes[1]
    rel_m = np.array([next(float(r["split_half_mean"]) for r in reliab if r["protocol"] == p) for p in PROTOCOLS])
    rel_lo = np.array([next(float(r["split_half_q025"]) for r in reliab if r["protocol"] == p) for p in PROTOCOLS])
    rel_hi = np.array([next(float(r["split_half_q975"]) for r in reliab if r["protocol"] == p) for p in PROTOCOLS])
    edge = [LONG, RAMP, BRIEF]
    ax.bar(
        xs, rel_m, 0.55, color=[soft_bar(c) for c in edge], edgecolor=edge, lw=0.9,
        yerr=np.vstack([rel_m - rel_lo, rel_hi - rel_m]), capsize=3,
        error_kw=dict(ecolor="#5A6570", lw=0.9), zorder=2,
    )
    ax.axhline(float(np.mean(obs_v)), ls="--", color=REF, lw=1.1, label="Mean cross-regime ρ (full n)", zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Long", "Ramp", "Brief"], fontsize=8)
    ax.set_ylabel("Split-half Spearman of S", color=INK)
    ylim_from_zero(ax, 1.05, pad=0)
    ax.legend(
        frameon=False, fontsize=7, loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
    )
    letter(ax, "B")
    save_fig(fig, "Figure 3-3")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def table_s1():
    summary = json.loads(PHENO_SUM.read_text())
    rows = []
    for p in PROTOCOLS:
        for feat in summary[p]["features"]:
            rows.append({"protocol": p, "short_name": SHORT[p], "feature": feat, "n_cells": summary[p]["n_cells"], "k80": summary[p]["k80"]})
    write_csv(TAB / "TableS1_features_protocols.csv", rows)


def table_s2():
    summary = json.loads(PHENO_SUM.read_text())
    q1 = load_csv(Q1)
    triple = None
    # from cross protocol
    rows = []
    for p in PROTOCOLS:
        rows.append(
            {
                "step": "phenotype_PCA",
                "protocol": p,
                "n_cells": summary[p]["n_cells"],
                "n_features": summary[p]["n_features"],
                "k80": summary[p]["k80"],
                "note": "",
            }
        )
    for r in q1:
        if r["k_rule"] != "k80":
            continue
        rows.append(
            {
                "step": "cross_protocol_shared_cells",
                "protocol": r["pair"],
                "n_cells": r["n_shared"],
                "n_features": "",
                "k80": f"{r['k_a']}/{r['k_b']}",
                "note": f"Spearman_D={float(r['spearman_D']):.3f}",
            }
        )
    # join coverage
    join = load_csv(JOIN)
    n_t = sum(1 for r in join if (r.get("t_type") or "").strip())
    rows.append({"step": "molecular_join", "protocol": "all", "n_cells": len(join), "n_features": "", "k80": "", "note": f"n_with_t_type={n_t}"})
    split = json.loads((Q1_REL / "split_half_summary.json").read_text()) if (Q1_REL / "split_half_summary.json").exists() else {}
    wgcna = json.loads((Q3_MOB / "wgcna_summary.json").read_text()) if (Q3_MOB / "wgcna_summary.json").exists() else {}
    rows.append(
        {
            "step": "split_half_long_square",
            "protocol": "long_pulse",
            "n_cells": split.get("n_both_halves_ok", 3598),
            "n_features": "",
            "k80": "11",
            "note": "both halves succeeded; Statistical Table u; Fig 1-3",
        }
    )
    rows.append(
        {
            "step": "mobility_triple_shared",
            "protocol": "all",
            "n_cells": wgcna.get("n_cells_mobility", 4049),
            "n_features": "",
            "k80": "",
            "note": "mean=0.589; sd=0.110; Statistical Table x; Fig 4-2",
        }
    )
    rows.append(
        {
            "step": "wgcna_rna_join",
            "protocol": "all",
            "n_cells": wgcna.get("n_cells_used", 3906),
            "n_features": "",
            "k80": "",
            "note": "2000-gene panel; unsigned WGCNA; Fig 4-4",
        }
    )
    rows.append(
        {
            "step": "wgcna_ttype_residual",
            "protocol": "all",
            "n_cells": 3902,
            "n_features": "",
            "k80": "",
            "note": "types with >=3 cells; Table 2 y; Extended Data Figure 4-3",
        }
    )
    write_csv(TAB / "TableS2_sample_counts.csv", rows)


def table_s3():
    rows = load_csv(TYPES)
    write_csv(TAB / "TableS3_ttype_counts.csv", rows)
    # also Cre summary from within_cre
    within = load_csv(Q2W)
    cre_rows = []
    for r in within:
        cre_rows.append(
            {
                "protocol": r["protocol"],
                "cre_line": r["cre_line"],
                "n_cells": r["n_cells"],
                "n_t_types": r["n_t_types"],
                "A_spearman": r["A_spearman"],
                "empirical_p": r["empirical_p"],
                "eta2": r["eta2"],
            }
        )
    write_csv(TAB / "TableS3b_within_cre.csv", cre_rows)


def table_s4():
    """Master statistics table."""
    rows = []
    # Q1
    for r in load_csv(Q1):
        if r["k_rule"] != "k80":
            continue
        rows.append(
            {
                "question": "Q1",
                "test": "Spearman_D",
                "contrast": r["pair"],
                "n": r["n_shared"],
                "effect": f"{float(r['spearman_D']):.4f}",
                "null_mean": f"{float(r['null_spearman_mean']):.4f}",
                "p_or_emp": f"{float(r['empirical_p']):.4f}",
                "extra": f"Jaccard={float(r['knn_jaccard_mean']):.4f}",
            }
        )
    # Q2
    for r in load_csv(Q2):
        for key, label in [
            ("A_t_type", "A_t_type"),
            ("A_cre", "A_Cre"),
            ("A_hier", "A_hier"),
            ("A_t_type_after_cre_residual", "A_t|Cre"),
            ("eta2_t_type", "eta2_t_type"),
            ("eta2_cre", "eta2_Cre"),
            ("eta2_t_type_after_cre_residual", "eta2_t|Cre"),
            ("knn_t_type_enrichment", "knn_enrichment"),
        ]:
            pkey = key + "_p" if key.startswith("A_") else ""
            rows.append(
                {
                    "question": "Q2",
                    "test": label,
                    "contrast": r["protocol"],
                    "n": r["n_with_t_type"],
                    "effect": f"{float(r[key]):.4f}",
                    "null_mean": r.get("A_t_type_null", "") if key == "A_t_type" else "",
                    "p_or_emp": r.get(pkey, "") if pkey in r else "",
                    "extra": f"k={r['k']}",
                }
            )
    # Q3a — manuscript / Statistical Table j–l use the 1000 type-label shuffles
    pair_key = {
        ("long_pulse", "ramp"): "long_ramp",
        ("long_pulse", "brief_pulse"): "long_brief",
        ("ramp", "brief_pulse"): "ramp_brief",
    }
    letter = {"long_ramp": "j", "long_brief": "l", "ramp_brief": "k"}
    conc1000 = {r["pair"]: r for r in load_csv(CONC_1000)} if CONC_1000.exists() else {}
    for r in load_csv(CONC):
        key = pair_key[(r["protocol_a"], r["protocol_b"])]
        r1k = conc1000.get(key)
        rows.append(
            {
                "question": "Q3a",
                "test": "Spearman_S",
                "contrast": f"{r['protocol_a']}__{r['protocol_b']}",
                "n": r["n_type_pairs"],
                "effect": f"{float((r1k or r).get('observed_rho', r['spearman_rho'])):.4f}",
                "null_mean": f"{float((r1k or r).get('null_mean', r['null_mean'])):.4f}",
                "p_or_emp": f"{float((r1k or r).get('empirical_p', r['empirical_p_vs_shuffle'])):.6f}".rstrip("0").rstrip(".")
                if r1k
                else f"{float(r['empirical_p_vs_shuffle']):.4f}",
                "extra": f"n_types={r['n_types']}; 1000 type-label shuffles; Statistical_Table={letter[key]}"
                if r1k
                else f"n_types={r['n_types']}",
            }
        )
    tags = Counter(r["tag"] for r in load_csv(PAIRS))
    rows.append(
        {
            "question": "Q3a",
            "test": "pair_tags",
            "contrast": "all_pairs",
            "n": str(sum(tags.values())),
            "effect": f"stable={tags['stable']}; intermediate={tags['intermediate']}; selective={tags['selective']}",
            "null_mean": "",
            "p_or_emp": "",
            "extra": f"stable_cv<{STABLE_RANK_CV}; selective_range>={SELECTIVE_RANK_RANGE}",
        }
    )
    # Q3b — primary panel in the manuscript uses the 1000-shuffle file (Statistical Table m–n)
    primary_1000 = {r["contrast"]: r for r in load_csv(Q3B_PRIMARY_1000)} if Q3B_PRIMARY_1000.exists() else {}
    for r in load_csv(Q3B):
        is_primary = r["panel"] == "primary" and primary_1000
        mean_null = primary_1000["T_vs_mean_S"] if is_primary else None
        range_null = primary_1000["T_vs_rank_range"] if is_primary else None
        rows.append(
            {
                "question": "Q3b",
                "test": "rho_T_meanS",
                "contrast": r["panel"],
                "n": r["n_pairs"],
                "effect": f"{float(r['spearman_T_vs_meanS']):.4f}",
                "null_mean": f"{float(mean_null['null_mean']):.4f}" if mean_null else "",
                "p_or_emp": f"{float(mean_null['emp_p_one_sided']):.6f}".rstrip("0").rstrip(".") if mean_null else f"{float(r['empirical_p_T_vs_meanS']):.4f}",
                "extra": f"n_genes={r['n_genes']}; 1000 shuffles; Statistical_Table=m" if mean_null else f"n_genes={r['n_genes']}",
            }
        )
        rows.append(
            {
                "question": "Q3b",
                "test": "rho_T_rank_range",
                "contrast": r["panel"],
                "n": r["n_pairs"],
                "effect": f"{float(r['spearman_T_vs_rank_range']):.4f}",
                "null_mean": f"{float(range_null['null_mean']):.4f}" if range_null else "",
                "p_or_emp": f"{float(range_null['emp_p_one_sided']):.6f}".rstrip("0").rstrip(".") if range_null else f"{float(r['empirical_p_T_vs_rank_range']):.4f}",
                "extra": f"n_genes={r['n_genes']}; 1000 shuffles; Statistical_Table=n" if range_null else f"n_genes={r['n_genes']}",
            }
        )
    # Q3c
    for r in load_csv(MODS):
        rows.append(
            {
                "question": "Q3c",
                "test": "module_D_vs_meanS",
                "contrast": f"{r['module']}[{r['role']}]",
                "n": r["n_genes_present"],
                "effect": f"{float(r['spearman_D_vs_meanS']):.4f}",
                "null_mean": "",
                "p_or_emp": f"{float(r['empirical_p_D_vs_meanS']):.4f}",
                "extra": f"vs_rewrite={float(r['spearman_D_vs_rank_range']):.4f}",
            }
        )
    # Statistical Table u–y (locked rounding: within_adjusted 0.106543 → 0.107, not 0.106)
    rows.extend(
        [
            {
                "question": "Q1",
                "test": "split_half_15NN_Jaccard",
                "contrast": "halfA_vs_halfB",
                "n": "3598",
                "effect": "0.109",
                "null_mean": "0.0026",
                "p_or_emp": "",
                "extra": "obs=0.109; chance=0.0026; adjusted=0.107; Statistical_Table=u; 100 index shuffles",
            },
            {
                "question": "Q1",
                "test": "split_half_cross_regime_15NN_Jaccard",
                "contrast": "long_ramp__long_brief__ramp_brief",
                "n": "3276 / 3509 / 3351",
                "effect": "0.025 / 0.024 / 0.020",
                "null_mean": "",
                "p_or_emp": "",
                "extra": "0.025 / 0.024 / 0.020 (long-ramp / long-brief / ramp-brief); n=3276/3509/3351 within the 3598-cell split-half pool; Statistical_Table=v",
            },
            {
                "question": "Q1",
                "test": "knn_Jaccard_k600_chance_adjusted",
                "contrast": "long_ramp__long_brief__ramp_brief",
                "n": "",
                "effect": "0.134",
                "null_mean": "",
                "p_or_emp": "",
                "extra": "at k=600: 0.134 / 0.101 / 0.108 (long-ramp / long-brief / ramp-brief); n=4054/4336/4134; Statistical_Table=w; 100 index shuffles",
            },
            {
                "question": "Q3",
                "test": "mobility",
                "contrast": "triple_shared",
                "n": "4049",
                "effect": "0.589",
                "null_mean": "",
                "p_or_emp": "",
                "extra": "mean=0.589; sd=0.110; Statistical_Table=x",
            },
            {
                "question": "Q3",
                "test": "WGCNA_ME_vs_mobility_ttype_residual",
                "contrast": "all_modules",
                "n": "3902",
                "effect": "0.039",
                "null_mean": "",
                "p_or_emp": "",
                "extra": "residual |rho|<=0.039; no module with |rho|>=0.10 and emp p<0.05; Statistical_Table=y; 1000 label shuffles",
            },
        ]
    )
    write_csv(TAB / "TableS4_all_statistics.csv", rows)


def table_s6():
    """Extended Data Figure 4-3 (Word table): WGCNA module vs mobility."""
    src = Q3_MOB / "wgcna_module_mobility_association.csv"
    if not src.exists():
        return
    assoc = sorted(load_csv(src), key=lambda r: abs(float(r["spearman_unadjusted"])), reverse=True)
    machine = []
    word = []
    for r in assoc:
        machine.append(
            {
                "module": r["module"],
                "n_genes": r["n_genes"],
                "hub": r["hub_gene"],
                "spearman_unadjusted": r["spearman_unadjusted"],
                "empirical_p_unadjusted": r["empirical_p_unadjusted"],
                "spearman_ttype_residual": r["spearman_ttype_residual"],
                "empirical_p_ttype_residual": r["empirical_p_residual"],
            }
        )
        word.append(
            {
                "Module": r["module"],
                "n genes": r["n_genes"],
                "Hub": r["hub_gene"],
                "Spearman ρ unadjusted": f"{float(r['spearman_unadjusted']):.3f}",
                "Empirical p unadjusted": f"{float(r['empirical_p_unadjusted']):.3f}",
                "Spearman ρ t-type residual": f"{float(r['spearman_ttype_residual']):.3f}",
                "Empirical p t-type residual": f"{float(r['empirical_p_residual']):.3f}",
            }
        )
    write_csv(TAB / "TableS6_wgcna_mobility.csv", machine)
    write_csv(TAB / "Table 4-3.csv", word)


def table_s5():
    rows = []
    for r in load_csv(MODS):
        rows.append(
            {
                "module": r["module"],
                "role": r["role"],
                "n_genes_present": r["n_genes_present"],
                "genes": r["genes"],
                "spearman_D_vs_meanS": r["spearman_D_vs_meanS"],
                "spearman_D_vs_rank_range": r["spearman_D_vs_rank_range"],
            }
        )
    write_csv(TAB / "TableS5_module_definitions.csv", rows)


def write_readme():
    (FIG / "README.md").write_text(
        "# eNeuro supplement figures\n\n"
        "Lock: `docs/SUPPLEMENT.md`.\n\n"
        "| File | Content |\n"
        "|------|----------|\n"
        "| `Figure 1-1` | Features · scree · k₈₀ |\n"
        "| `Figure 1-2` | k₈₀ vs 5-PC D · kNN scale sweep · PC1 loadings |\n"
        "| `Figure 1-3` | Split-half within-regime vs cross-regime 15-NN Jaccard |\n"
        "| `Figure 2-1` | A_p · within-Cre |\n"
        "| `Figure 3-2` | Full S matrices · tag thresholds · subclass |\n"
        "| `Figure 3-3` | Centroid / S concordance robustness |\n"
        "| `Figure 4-2` | 3 RNA panels · marker controls |\n"
        "| `Figure 4-4` | Mobility distribution · WGCNA module–mobility ρ ± t-type residual |\n\n"
        "Tables: `tables/supplement/`.\n"
        "Generated by `scripts/figures_eneuro_supplement.py`.\n"
    )
    (TAB / "README.md").write_text(
        "# eNeuro supplement tables\n\n"
        "| File | Content |\n"
        "|------|----------|\n"
        "| `TableS1_features_protocols.csv` | Feature lists per regime (**Table 1-3**; submitted — do not reuse this number) |\n"
        "| `TableS2_sample_counts.csv` | n / joins / exclusions notes (includes split-half 3598, mobility 4049, WGCNA 3906, residual 3902) |\n"
        "| `TableS3_ttype_counts.csv` | t-type counts |\n"
        "| `TableS3b_within_cre.csv` | Within-Cre A / η² |\n"
        "| `TableS4_all_statistics.csv` | Master stats + nulls (Table 2 a–y; Word inventory is Extended Data Table 2-1) |\n"
        "| `TableS5_module_definitions.csv` | Module gene sets (Extended Data Figure 4-1) |\n"
        "| `TableS6_wgcna_mobility.csv` | WGCNA module–mobility ρ ± t-type residual (machine precision) |\n"
        "| `Table 4-3.csv` | Same as TableS6, Word-facing rounding; Extended Data Figure 4-3 (related to Figure 4-4). |\n"
    )


def main():
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    print("Fig S1…")
    fig_s1()
    print("Fig S2…")
    fig_s2()
    print("Fig 1-3…")
    fig_s13()
    print("Fig S3…")
    fig_s3()
    print("Fig S4…")
    fig_s4()
    print("Fig S5…")
    fig_s5()
    print("Fig 3-3…")
    fig_s32()
    print("Fig 4-4…")
    fig_mobility()
    print("Tables…")
    table_s1()
    table_s2()
    table_s3()
    table_s4()
    table_s5()
    table_s6()
    write_readme()
    print("done →", FIG, TAB)


if __name__ == "__main__":
    main()
