#!/usr/bin/env python3
"""Four eNeuro Results figures → figures/results/

Q1 DESCRIBE · Q2 ANCHOR · Q3 COMPARE · Q4 EXPLAIN
No on-figure paper titles (eNeuro: title belongs in the legend).
Muted Blue / Orange / Green protocol palette.
"""

from __future__ import annotations

import colorsys
import csv
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-prg")

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

REV = Path(__file__).resolve().parents[1]
REPO = REV.parent
OUT = REV / "figures/results"
PHENO = REPO / "results/phenotype"
JOIN = REPO / "results/m0/m0_molecular_join.csv"
Q1 = REPO / "results/q1/q1_cross_protocol.csv"
Q1_JAC = REV / "results/q1/q1_knn_jaccard_k15.csv"
Q1_SCALE = REV / "results/q1/q1_knn_scale.csv"
Q2 = REPO / "results/q2/q2_anchor.csv"
PAIRS = REPO / "results/q3a/q3a_pair_tags.csv"
CONC = REPO / "results/q3a/q3a_S_concordance.csv"
TPAIRS = REPO / "results/q3b/q3b_pairs_with_T_primary.csv"
MODS = REPO / "results/q3c/q3c_module_tests.csv"
NWB = REPO / "data/official/dandi_000020/sub-639391169/sub-639391169_ses-639797293_icephys.nwb"
TRACE_CSV = REPO / "results/fig1_example_traces.csv"

# A = muted regime hues (not neon). B/C pairs = darker, hue-shifted (slate / teal / umber).
LONG, RAMP, BRIEF = "#3B8FB0", "#C4843C", "#2A9A74"
INK, MUTED = "#3D4A55", "#9AA3AD"
PCOL = {"long_pulse": LONG, "ramp": RAMP, "brief_pulse": BRIEF}
SHORT = {"long_pulse": "Long", "ramp": "Ramp", "brief_pulse": "Brief"}
PROTOCOLS = ["long_pulse", "ramp", "brief_pulse"]
TAG = {"stable": "#6E8B9A", "selective": "#B88484", "intermediate": "#C5B6A4"}
TAG_HATCH = {"stable": "//", "intermediate": "o", "selective": "xxxx"}
SAND, HIER, REF = "#B8A07A", "#6A8B84", "#5A6570"
# Q2 label layer — locked: t-type gray, Cre brown, hier pale black, residual mix (not regime / pair / Fig. 4)
LAB_T, LAB_CRE, LAB_HIER = "#6B7280", "#8B6B4A", "#4A4A4A"
LAB_T_CRE = mcolors.to_hex(
    tuple((a + b) / 2 for a, b in zip(mcolors.to_rgb(LAB_T), mcolors.to_rgb(LAB_CRE)))
)

PAIR = "#7A858C"
PAIR_FILL = 0.38
# Parent family kept, but hue-shifted and darkened so pairs do not sit on regime colors:
# Long (dusty cyan) → deep slate; Brief (sea-green) → deep teal; Ramp (muted apricot) → umber.
PAIR_LR, PAIR_LB, PAIR_RB = "#1F3F58", "#0E4A47", "#542616"
PAIR_COL = {
    "long_pulse__ramp": PAIR_LR,
    "long_pulse__brief_pulse": PAIR_LB,
    "ramp__brief_pulse": PAIR_RB,
}


PAIR_SPECS = [
    ("long_pulse", "ramp", "long_pulse__ramp", "Long–Ramp"),
    ("long_pulse", "brief_pulse", "long_pulse__brief_pulse", "Long–Brief"),
    ("ramp", "brief_pulse", "ramp__brief_pulse", "Ramp–Brief"),
]
SUBCOL = {
    "Pvalb": "#B88484",
    "Sst": "#6E8B9A",
    "Vip": "#8B7E9A",
    "Lamp5": "#7A8FA0",
    "Sncg": "#B89A6E",
    "Serpinf1": "#A39E96",
}
MOD = {"shared": "#7A6B8A", "rewrite": "#B88484"}
SCORE_FILES = {p: PHENO / f"{p}_pca_scores.csv" for p in PROTOCOLS}
VAR_FILES = {p: PHENO / f"{p}_pca_variance.csv" for p in PROTOCOLS}


def soft_cmap(hex_color: str, name: str = "soft"):
    c = mcolors.hex2color(hex_color)
    return mcolors.LinearSegmentedColormap.from_list(
        name,
        [(1, 1, 1), tuple(0.72 + 0.28 * x for x in c), tuple(0.35 + 0.65 * x for x in c), c],
    )


def pair_cmap(color, name: str = "pair"):
    """Fig. 1B hexbin: washed tints of the canonical pair color (pale, not vivid)."""
    c = mcolors.to_rgb(color)

    def mix(t):
        return tuple((1.0 - t) + t * x for x in c)

    return mcolors.LinearSegmentedColormap.from_list(
        name, [mix(0.06), mix(0.18), mix(0.34), mix(0.50)]
    )


def hexbin_cmap(color, name: str = "hexbin"):
    """Fig. 4 A/B hexbin: same mix family as pair_cmap, deeper endpoint (not near-black)."""
    c = mcolors.to_rgb(color)

    def mix(t):
        return tuple((1.0 - t) + t * x for x in c)

    return mcolors.LinearSegmentedColormap.from_list(
        name, [mix(0.08), mix(0.24), mix(0.44), mix(0.65)]
    )


CMAP_HEAT = soft_cmap(LONG, "heat")
CMAP_RNA = soft_cmap("#6E8B9A", "rna")
CMAP_PROTO = {p: soft_cmap(PCOL[p], p) for p in PROTOCOLS}


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
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.edgecolor": MUTED,
            "xtick.color": "#5A6570",
            "ytick.color": "#5A6570",
            "axes.labelcolor": INK,
            "text.color": INK,
            "hatch.linewidth": 0.7,
            "pdf.fonttype": 42,
        }
    )


def letter(ax, s: str):
    ax.text(
        -0.10, 1.02, s, transform=ax.transAxes,
        fontweight="bold", fontsize=12, va="bottom", ha="left", clip_on=False,
    )


def letter_fig(ax, s: str, x=None):
    """Panel letter in figure coords so A/C can share one x."""
    bb = ax.get_position()
    if x is None:
        x = bb.x0 - 0.045
    y = bb.y1 + 0.004
    ax.figure.text(
        x, y, s, fontweight="bold", fontsize=12, va="bottom", ha="left",
    )
    return x


def save(fig, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=300, facecolor="white", pad_inches=0.12)
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white", pad_inches=0.12)
    plt.close(fig)
    print("wrote", OUT / stem)


def ylim_from_zero(ax, hi=None, pad=0.08):
    """Pin the y origin to 0; pad only above the data."""
    if hi is None:
        hi = ax.get_ylim()[1]
    hi = max(float(hi), 0.0)
    ax.set_ylim(0.0, hi * (1.0 + pad) if hi > 0 else 1.0)


def pc_cols(names):
    cols = [c for c in names if c.startswith("PC") and c[2:].isdigit()]
    return sorted(cols, key=lambda c: int(c[2:]))


def load_scores(path: Path):
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


def load_csv(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_join(path: Path):
    with open(path, newline="") as f:
        return {row["file_stem"]: row for row in csv.DictReader(f)}


def load_k80(path: Path) -> int:
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if float(row["cumulative_explained_variance_ratio"]) >= 0.80:
                return int(float(row["component"]))
    return 1


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def pairwise_euclidean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    nrm = np.sum(x * x, axis=1, keepdims=True)
    d2 = nrm + nrm.T - 2.0 * np.dot(x, x.T)
    np.maximum(d2, 0.0, out=d2)
    d = np.sqrt(d2)
    np.fill_diagonal(d, 0.0)
    return d


def aligned_pair(pack, a, b):
    pos_a = {s: i for i, s in enumerate(pack[a]["stems"])}
    shared = [s for s in pack[b]["stems"] if s in pos_a]
    pos_b = {s: i for i, s in enumerate(pack[b]["stems"])}
    xa = pack[a]["X"][np.array([pos_a[s] for s in shared]), : pack[a]["k"]]
    xb = pack[b]["X"][np.array([pos_b[s] for s in shared]), : pack[b]["k"]]
    return shared, xa, xb


def sample_pair_distances(xa, xb, rng, n_pairs=40000):
    da = pairwise_euclidean(xa)
    db = pairwise_euclidean(xb)
    n = da.shape[0]
    i = rng.integers(0, n, size=n_pairs * 3)
    j = rng.integers(0, n, size=n_pairs * 3)
    m = i < j
    i, j = i[m][:n_pairs], j[m][:n_pairs]
    return da[i, j], db[i, j]


def hex_axis_hi(d):
    """Use the dense cloud, not the long tail, as the hexbin window."""
    d = np.asarray(d, dtype=float)
    return float(np.percentile(d, 97.5)) * 1.02


def snap_hex_hi(h, n_ticks=5):
    """Nearest nice 5-tick limit; prefer a small clip over empty padding."""
    if h <= 0:
        return 1.0
    n_step = n_ticks - 1
    cands = {
        m * (10.0 ** k) * n_step
        for k in range(-1, 3)
        for m in (1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
    }
    scored = []
    for c in cands:
        if c < 0.88 * h or c > 1.10 * h:
            continue
        clip = max(0.0, (h - c) / h)
        pad = max(0.0, (c - h) / h)
        scored.append((pad * 1.4 + clip, c))
    if scored:
        return float(sorted(scored)[0][1])
    return float(min(cands, key=lambda c: abs(c - h)))


def hex_ticks(hi, n=5):
    t = np.linspace(0.0, hi, n)
    labels = []
    for v in t:
        if abs(v - round(v)) < 0.05:
            labels.append(f"{int(round(v))}")
        else:
            labels.append(f"{v:.1f}".rstrip("0").rstrip("."))
    return t, labels


def subclass_of(t: str) -> str:
    return t.split()[0] if t else ""


def short_type(t: str) -> str:
    p = t.split()
    return t if len(p) <= 2 else f"{p[0]} {p[-1]}"


def genotype(rec):
    return (rec.get("full_genotype") or rec.get("cre_line") or "").strip()


def is_cre_driver(rec, driver: str) -> bool:
    g = genotype(rec)
    return g == driver or g.startswith(driver + "/") or g.startswith(driver + ";")


def kde_1d(x, grid):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.zeros_like(grid)
    bw = max(1.06 * float(x.std()) * (x.size ** -0.2), 1e-3)
    z = (grid[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * z * z).sum(axis=1) / (x.size * bw * np.sqrt(2 * np.pi))


def soft_bar_color(hex_color: str, alpha: float = 0.42):
    """Solid color matching a translucent fill over white (as in density panels)."""
    r, g, b = mcolors.to_rgb(hex_color)
    return (1 - alpha + alpha * r, 1 - alpha + alpha * g, 1 - alpha + alpha * b)


def quintile_transfer(ax, src, dst, vmin=0.0, vmax=0.65):
    def qbin(v):
        r = np.argsort(np.argsort(v))
        return np.minimum(4, (5 * r / max(1, len(v) - 1)).astype(int))

    a, b = qbin(src), qbin(dst)
    M = np.zeros((5, 5))
    for i, j in zip(a, b):
        M[i, j] += 1
    P = M / np.maximum(M.sum(1, keepdims=True), 1)
    im = ax.imshow(P, origin="lower", cmap=CMAP_HEAT, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(5):
        for j in range(5):
            ax.text(
                j,
                i,
                f"{P[i, j]:.0%}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if P[i, j] > 0.36 else INK,
            )
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(list("12345"))
    ax.set_yticklabels(list("12345"))
    for s in ax.spines.values():
        s.set_visible(True)
    return P, im


def load_trace_csv(path: Path):
    by = {p: ([], []) for p in PROTOCOLS}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            p = row.get("protocol") or ""
            if p not in by:
                continue
            by[p][0].append(float(row["t"]))
            by[p][1].append(float(row["v"]))
    out = []
    for p in PROTOCOLS:
        t, v = by[p]
        if not t:
            return None
        out.append((np.asarray(t, dtype=float), np.asarray(v, dtype=float), p))
    return out


def load_traces():
    """Example Long / Ramp / Brief sweeps for Figure 1A (DANDI session 639797293).

    Uses the bundled CSV. The source NWB is only read if that CSV is missing.
    """
    if TRACE_CSV.exists():
        loaded = load_trace_csv(TRACE_CSV)
        if loaded:
            return loaded
    if not NWB.exists():
        return None
    os.environ.setdefault("PYNWB_HOME", str(REPO / "data/official/.pynwb"))
    from ipfx.dataset.create import create_ephys_data_set
    from ipfx.sweep import Sweep

    orig = Sweep.__init__

    def patched(self, t, v, i, clamp_mode, sampling_rate, sweep_number=None, epochs=None):
        if epochs:
            epochs = {k: val for k, val in epochs.items() if val is not None}
        orig(self, t, v, i, clamp_mode, sampling_rate, sweep_number=sweep_number, epochs=epochs)

    Sweep.__init__ = patched
    ds = create_ephys_data_set(nwb_file=str(NWB))

    def pack(sn):
        sw = ds.sweep(int(sn))
        t = np.asarray(sw._t) - np.asarray(sw._t)[0]
        return t, np.asarray(sw._v)

    ls_t, ls_v = pack(14)
    rp_t, rp_v = pack(34)
    ss_t, ss_v = pack(26)
    ls_m = (ls_t >= 0.7) & (ls_t <= 2.4)
    t0 = rp_t[np.where(rp_v > -10)[0][0]]
    rp_m = (rp_t >= t0 - 1.5) & (rp_t <= t0 + 2.5)
    t1 = ss_t[np.where(ss_v > -10)[0][0]]
    ss_m = (ss_t >= t1 - 0.008) & (ss_t <= t1 + 0.020)
    return [
        (ls_t[ls_m], ls_v[ls_m], "long_pulse"),
        (rp_t[rp_m], rp_v[rp_m], "ramp"),
        (ss_t[ss_m], ss_v[ss_m], "brief_pulse"),
    ]


# ---------------------------------------------------------------------------
# Fig 1 — DESCRIBE
# ---------------------------------------------------------------------------
def fig1(pack, q1_rows, traces, jac_rows, scale_rows):
    """A traces · B global pairwise D · C local 15-NN Jaccard."""
    fig = plt.figure(figsize=(12.6, 10.4))
    outer = gridspec.GridSpec(
        3, 1, height_ratios=[0.85, 1.25, 1.05],
        hspace=0.42, left=0.07, right=0.98, top=0.96, bottom=0.07,
    )
    row0 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.28)
    row1 = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[1], width_ratios=[1.0, 1.0, 1.0, 0.70], wspace=0.28,
    )
    row2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], wspace=0.28)
    rng = np.random.default_rng(0)

    if traces:
        for i, (t, v, p) in enumerate(traces):
            ax = fig.add_subplot(row0[0, i])
            ax.plot(t, v, color=PCOL[p], lw=0.85)
            ax.set_title(SHORT[p], fontsize=11, fontweight="bold", color=PCOL[p], pad=4)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("mV")
            if i == 0:
                letter(ax, "A")
    else:
        for i, p in enumerate(PROTOCOLS):
            ax = fig.add_subplot(row0[0, i])
            ax.text(0.5, 0.5, f"{SHORT[p]}\n(trace missing)", ha="center", va="center", transform=ax.transAxes)
            if i == 0:
                letter(ax, "A")

    k80 = [r for r in q1_rows if r["k_rule"] == "k80"]
    k80 = {r["pair"]: r for r in k80}
    chance_k15 = {
        r["pair"]: float(r["chance_jaccard_mean"])
        for r in scale_rows
        if int(float(r["k"])) == 15
    }

    hex_panels = []
    for a, b, key, lab in PAIR_SPECS:
        _, xa, xb = aligned_pair(pack, a, b)
        ua, ub = sample_pair_distances(xa, xb, rng)
        hex_panels.append(
            (a, b, key, lab, ua, ub, snap_hex_hi(hex_axis_hi(ua)), snap_hex_hi(hex_axis_hi(ub)))
        )

    for i, (a, b, key, lab, ua, ub, xmax, ymax) in enumerate(hex_panels):
        ax = fig.add_subplot(row1[0, i])
        cmap = pair_cmap(PAIR_COL[key], f"d{i}")
        hb = ax.hexbin(
            ua, ub, gridsize=32, cmap=cmap, mincnt=1, linewidths=0, rasterized=True,
            extent=(0, xmax, 0, ymax),
        )
        counts = np.asarray(hb.get_array(), dtype=float)
        counts = counts[np.isfinite(counts) & (counts > 0)]
        if counts.size:
            hb.set_clim(1, max(2.0, float(np.percentile(counts, 90))))
        xt, xl = hex_ticks(xmax)
        yt, yl = hex_ticks(ymax)
        ax.set_xlim(-0.5 * (xt[1] - xt[0]), xmax + 0.5 * (xt[1] - xt[0]))
        ax.set_ylim(-0.5 * (yt[1] - yt[0]), ymax + 0.5 * (yt[1] - yt[0]))
        ax.set_xticks(xt)
        ax.set_xticklabels(xl)
        ax.set_yticks(yt)
        ax.set_yticklabels(yl)
        ax.set_xlabel(f"{SHORT[a]} D")
        ax.set_ylabel(f"{SHORT[b]} D")
        ax.set_title(lab, fontsize=10, fontweight="bold", color=PAIR_COL[key], pad=3)
        rho = float(k80[key]["spearman_D"])
        ax.text(
            0.05, 0.95, f"ρ = {rho:.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, color=INK, fontweight="bold",
        )
        if i == 0:
            letter(ax, "B")

    ax = fig.add_subplot(row1[0, 3])
    xs = np.arange(3)
    obs = [float(k80[key]["spearman_D"]) for *_, key, _ in PAIR_SPECS]
    pair_edges = [PAIR_COL[key] for *_, key, _ in PAIR_SPECS]
    ax.bar(
        xs, obs, 0.48,
        color=[soft_bar_color(c, PAIR_FILL) for c in pair_edges],
        edgecolor=pair_edges, lw=0.8, zorder=2,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(["Long–\nRamp", "Long–\nBrief", "Ramp–\nBrief"], fontsize=7.5)
    ax.set_ylabel("Spearman of cell–cell D")
    ylim_from_zero(ax, 0.50, pad=0.10)
    for x, y in zip(xs, obs):
        ax.text(x, y + 0.015, f"{y:.3f}", ha="center", va="bottom", fontsize=8)

    jac_by = {}
    for r in jac_rows:
        jac_by.setdefault(r["pair"], []).append(float(r["jaccard_k15"]))
    c_axes = []
    y_hi = 0.0
    for i, (a, b, key, lab) in enumerate(PAIR_SPECS):
        ax = fig.add_subplot(row2[0, i])
        vals = np.asarray(jac_by.get(key, []), dtype=float)
        if vals.size:
            bin_w = 0.05
            edges = np.arange(-bin_w / 2.0, 0.175 + 1e-9, bin_w)
            counts, _ = np.histogram(vals, bins=edges)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.bar(
                centers, counts, width=bin_w * 0.72, align="center",
                color=soft_bar_color(PAIR_COL[key], PAIR_FILL),
                edgecolor=PAIR_COL[key], lw=0.6, zorder=2,
            )
            y_hi = max(y_hi, float(counts.max()) if counts.size else 0.0)
        ax.set_xlim(-0.03, 0.175)
        ax.set_xticks([0, 0.05, 0.10, 0.15])
        ax.set_xlabel("15-NN Jaccard")
        ax.set_ylabel("Cells")
        if i == 0:
            letter(ax, "C")
        ax.set_title(lab, fontsize=10, fontweight="bold", color=PAIR_COL[key], pad=3)
        if vals.size:
            ch = chance_k15.get(key)
            note = f"mean {np.mean(vals):.3f}"
            if ch is not None:
                note += f"  ·  chance {ch:.3f}"
            ax.text(0.98, 0.94, note, transform=ax.transAxes,
                    ha="right", va="top", fontsize=7.5, color=REF)
        c_axes.append(ax)
    if y_hi > 0:
        for ax in c_axes:
            ylim_from_zero(ax, y_hi, pad=0.08)
    save(fig, "Fig1_describe")


# ---------------------------------------------------------------------------
# Fig 2 — ANCHOR
# ---------------------------------------------------------------------------
def rain_horizontal(ax, groups, face, edge, rng, point_alpha=0.35, max_points=120):
    """Horizontal violin + black box + jittered points (reference raincloud style)."""
    data = [np.asarray(v, dtype=float)[np.isfinite(v)] for _, v in groups]
    data = [v for v in data if v.size]
    if not data:
        return
    n = len(data)
    parts = ax.violinplot(
        data, positions=np.arange(1, n + 1), vert=False,
        showmeans=False, showextrema=False, showmedians=False, widths=0.78,
    )
    for body in parts["bodies"]:
        body.set_facecolor(face)
        body.set_edgecolor(edge)
        body.set_alpha(0.55)
        body.set_linewidth(0.8)
    bp = ax.boxplot(
        data, positions=np.arange(1, n + 1), vert=False, widths=0.14,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", lw=1.4),
        boxprops=dict(facecolor="white", edgecolor="black", lw=1.1),
        whiskerprops=dict(color="black", lw=1.0),
        capprops=dict(color="black", lw=1.0),
        zorder=4,
    )
    for i, v in enumerate(data, start=1):
        show = v
        if len(show) > max_points:
            show = rng.choice(show, size=max_points, replace=False)
        # jitter above the midline so box stays readable
        jitter = rng.uniform(0.08, 0.28, size=len(show))
        ax.scatter(
            show, np.full(len(show), i) + jitter,
            s=11, c=edge, alpha=point_alpha, linewidths=0, zorder=3, rasterized=True,
        )
    ax.set_yticks(range(1, n + 1))
    ax.set_yticklabels([lab for lab, _ in groups[:n]], fontsize=8)
    ax.set_ylim(0.4, n + 0.75)
    return bp


def fig2_2x2(pack, join, q2_rows):
    """Figure 2: occupancy, within-Cre, η², kNN enrichment."""
    fig = plt.figure(figsize=(12.6, 9.8))
    # Nested right column so B and D share the same axes box (aligned plot frames)
    gs = gridspec.GridSpec(
        2, 2,
        hspace=0.42, wspace=0.34,
        left=0.11, right=0.96, top=0.86, bottom=0.08,
    )
    q2 = {r["protocol"]: r for r in q2_rows}
    soft_long = soft_bar_color(LONG, 0.38)
    soft_ramp = soft_bar_color(RAMP, 0.38)
    soft_brief = soft_bar_color(BRIEF, 0.38)
    rng = np.random.default_rng(0)

    # A — subclass occupancy: Long-tinted ridges + scatter
    ax = fig.add_subplot(gs[0, 0])
    p = "long_pulse"
    by = {s: [] for s in SUBCOL}
    for j, s in enumerate(pack[p]["stems"]):
        t = (join.get(s) or {}).get("t_type") or ""
        if subclass_of(t) in by:
            by[subclass_of(t)].append(pack[p]["X"][j, 0])
    series = [(s, np.array(by[s], dtype=float)) for s in SUBCOL if len(by[s]) >= 8]
    vals = [v[np.isfinite(v)] for _, v in series]
    all_v = np.concatenate(vals)
    lo, hi = np.percentile(all_v, [1, 99])
    pad = 0.06 * (hi - lo + 1e-6)
    grid = np.linspace(lo - pad, hi + pad, 220)
    dens = [kde_1d(v, grid) for v in vals]
    ymax = max(float(d.max()) for d in dens)
    scale = 0.86 / ymax
    for i, ((lab, v), d) in enumerate(zip(series, dens)):
        ax.fill_between(grid, i, i + d * scale, color=soft_long, alpha=0.95, lw=0, zorder=1)
        ax.plot(grid, i + d * scale, color=LONG, lw=1.25, zorder=2)
        med = float(np.median(v))
        ax.plot([med, med], [i, i + 0.72], color=REF, lw=1.2, ls="--", zorder=3)
        jitter = rng.normal(0, 0.06, size=len(v))
        ax.scatter(
            v, np.full(len(v), i + 0.22) + jitter,
            s=5, c=LONG, alpha=0.22, linewidths=0, zorder=4, rasterized=True,
        )
    ax.set_yticks([i + 0.28 for i in range(len(series))])
    ax.set_yticklabels([s[0] for s in series], fontsize=8)
    ax.set_xlabel("Long-pulse PC1")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(-0.15, len(series) - 0.05)
    letter(ax, "A")

    # B — within Cre t-types on the Long PC1 axis
    ax_b = fig.add_subplot(gs[0, 1])
    stems, X = pack["long_pulse"]["stems"], pack["long_pulse"]["X"]
    by_t = {}
    for j, s in enumerate(stems):
        rec = join.get(s) or {}
        if not is_cre_driver(rec, "Sst-IRES-Cre"):
            continue
        t = rec.get("t_type") or ""
        if t:
            by_t.setdefault(t, []).append(X[j, 0])
    top = sorted(by_t.items(), key=lambda kv: -len(kv[1]))[:6]
    groups_b = [(short_type(t), np.asarray(v, dtype=float)) for t, v in top]
    rain_horizontal(ax_b, groups_b, soft_long, LONG, rng, point_alpha=0.40, max_points=100)
    ax_b.set_xlabel("Long-pulse PC1")
    letter(ax_b, "B")

    # C — η² (label colors locked: gray / brown / mixed fill residual)
    ax = fig.add_subplot(gs[1, 0])
    xs = np.arange(3)
    w = 0.26
    eta_t = [float(q2[p]["eta2_t_type"]) for p in PROTOCOLS]
    eta_c = [float(q2[p]["eta2_cre"]) for p in PROTOCOLS]
    eta_r = [float(q2[p]["eta2_t_type_after_cre_residual"]) for p in PROTOCOLS]
    ax.bar(xs - w, eta_t, w, color=soft_bar_color(LAB_T, 0.42), edgecolor=LAB_T, lw=1.0, label="t-type", zorder=2)
    ax.bar(xs, eta_c, w, color=soft_bar_color(LAB_CRE, 0.50), edgecolor=LAB_CRE, lw=1.0, label="Cre", zorder=2)
    ax.bar(
        xs + w, eta_r, w, color=soft_bar_color(LAB_T_CRE, 0.55), edgecolor=LAB_T_CRE, lw=1.0,
        label="t-type after Cre", zorder=2,
    )
    ax.axhline(0, color="#5A6570", lw=0.8, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Long", "Ramp", "Brief"])
    ax.set_ylabel("η²")
    ylim_from_zero(ax, 0.50, pad=0)
    for i, (a, b, c) in enumerate(zip(eta_t, eta_c, eta_r)):
        ax.text(i - w, a + 0.01, f"{a:.2f}", ha="center", va="bottom", fontsize=7.5)
        ax.text(i, b + 0.01, f"{b:.2f}", ha="center", va="bottom", fontsize=7.5)
        ax.text(i + w, c + 0.01, f"{c:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    letter(ax, "C")

    # D — kNN: same column as B; normal bar xlim (aligned frame); short chance stub + label
    ax_d = fig.add_subplot(gs[1, 1])
    knn = [float(q2[p]["knn_t_type_enrichment"]) for p in PROTOCOLS]
    bar_w = 0.58
    ax_d.bar(xs, knn, bar_w, color=[soft_long, soft_ramp, soft_brief], edgecolor=[LONG, RAMP, BRIEF], lw=1.0, zorder=2)
    bar_right = float(xs[-1]) + bar_w / 2
    line_end = bar_right + 0.12
    ax_d.plot([-0.45, line_end], [1.0, 1.0], color=REF, lw=1.2, ls="--", zorder=5, solid_capstyle="butt")
    ax_d.text(line_end + 0.05, 1.0, "chance", ha="left", va="center", fontsize=7.5, color=MUTED, clip_on=False, zorder=6)
    ax_d.set_xticks(xs)
    ax_d.set_xticklabels(["Long", "Ramp", "Brief"])
    ax_d.set_ylabel("Same-type kNN\n/ chance")
    # Match a standard 3-bar frame (same visual width as B's data box in this column)
    ax_d.set_xlim(-0.55, 2.55)
    ylim_from_zero(ax_d, max(knn), pad=0.18)
    for x, y in zip(xs, knn):
        ax_d.text(x, y + 0.12, f"{y:.2f}×", ha="center", va="bottom", fontsize=9)
    letter(ax_d, "D")

    # Lock B/D to identical axes position so the plot frames align
    pos_b = ax_b.get_position()
    pos_d = ax_d.get_position()
    ax_d.set_position([pos_b.x0, pos_d.y0, pos_b.width, pos_d.height])

    save(fig, "Fig2_anchor_2x2")


# ---------------------------------------------------------------------------
# Fig 3 — COMPARE
# ---------------------------------------------------------------------------
def fig3_backbone(pairs, conc):
    """Figure 3: conserved S, pair tags, grouped trajectories."""
    for r in pairs:
        for k in ("rank_norm_long_pulse", "rank_norm_ramp", "rank_norm_brief_pulse"):
            r[k] = float(r[k])
    order = [("long_pulse", "ramp"), ("long_pulse", "brief_pulse"), ("ramp", "brief_pulse")]
    conc = sorted(conc, key=lambda r: order.index((r["protocol_a"], r["protocol_b"])))

    examples = [
        ("Pvalb Sema3e Kank4", "Vip Arhgap36 Hmcn1"),
        ("Pvalb Tpbg", "Vip Arhgap36 Hmcn1"),
        ("Pvalb Reln Itm2a", "Vip Crispld2 Htr2c"),
        ("Sst Chodl", "Vip Arhgap36 Hmcn1"),
        ("Sst Hpse Sema3c", "Sst Myh8 Etv1"),
        ("Pvalb Reln Tac1", "Sst Hpse Cbln4"),
    ]
    lookup = {(r["type_a"], r["type_b"]): r for r in pairs}
    lookup.update({(r["type_b"], r["type_a"]): r for r in pairs})
    rows = [(a, b, lookup[(a, b)]) for a, b in examples if (a, b) in lookup]
    rank = {"stable": 0, "intermediate": 1, "selective": 2}
    rows = sorted(rows, key=lambda t: (rank.get(t[2]["tag"], 9), t[0], t[1]))
    n = len(rows)

    fig = plt.figure(figsize=(12.6, 8.2))
    # Identical outer frame for both rows
    outer = gridspec.GridSpec(
        2, 1,
        height_ratios=[1.15, 1.15],
        hspace=0.42,
        left=0.20, right=0.84, top=0.94, bottom=0.08,
    )
    top = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0], width_ratios=[1.0, 1.35], wspace=0.30,
    )
    pair_keys = ["long_pulse__ramp", "long_pulse__brief_pulse", "ramp__brief_pulse"]
    pair_edges = [PAIR_COL[k] for k in pair_keys]

    # A — canonical pair hues (steel / teal / ochre)
    ax_a = fig.add_subplot(top[0, 0])
    xs = np.arange(3)
    rhos = [float(r["spearman_rho"]) for r in conc]
    ax_a.bar(
        xs, rhos, 0.58,
        color=[soft_bar_color(c, PAIR_FILL) for c in pair_edges],
        edgecolor=pair_edges, lw=1.0, zorder=2,
    )
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(["Long–Ramp", "Long–Brief", "Ramp–Brief"], fontsize=7.5)
    ax_a.set_ylabel("Spearman of type–type S")
    ylim_from_zero(ax_a, 0.85, pad=0)
    for x, y in zip(xs, rhos):
        ax_a.text(x, y + 0.025, f"{y:.3f}", ha="center", va="bottom", fontsize=8)
    ax_a.text(
        0.0, 1.04,
        "S = type–type phenotype distance map",
        transform=ax_a.transAxes, ha="left", va="bottom", fontsize=6.5, color=MUTED, clip_on=False,
    )

    # B — tag identity by hatch (slash / dots / grid), not a third hue
    ax = fig.add_subplot(top[0, 1])
    counts = Counter(r["tag"] for r in pairs)
    labs = ["stable", "intermediate", "selective"]
    vals = [counts[k] for k in labs]
    total = sum(vals) or 1
    pcts = [100.0 * v / total for v in vals]
    left = 0.0
    for lab, v, p in zip(labs, vals, pcts):
        ax.barh(
            0, v, left=left, height=0.48,
            color="#F4F6F7", edgecolor=INK, lw=0.9, hatch=TAG_HATCH[lab], zorder=2,
        )
        cx = left + v / 2
        ax.text(
            cx, 0.38, f"{lab}\n{v} ({p:.0f}%)",
            ha="center", va="bottom", fontsize=7.5, color=INK, linespacing=1.15,
        )
        left += v
    ax.set_yticks([])
    ax.set_xlim(0, total)
    ax.set_ylim(-0.45, 0.95)
    ax.set_xlabel("Type-pairs")
    ax.spines["left"].set_visible(False)
    letter(ax, "B")

    # C — one axes, same outer L/R as A|B
    ax = fig.add_subplot(outer[1])
    ylabels = []
    for i, (a, b, r) in enumerate(rows):
        y = n - 1 - i
        xs_r = [r["rank_norm_long_pulse"], r["rank_norm_ramp"], r["rank_norm_brief_pulse"]]
        tag = r["tag"]
        ax.plot([0, 1], [y, y], color="#ECEFF2", lw=11, solid_capstyle="round", zorder=1)
        ax.plot(xs_r, [y, y, y], color=MUTED, lw=1.6, alpha=0.55, zorder=2)
        ax.scatter(
            xs_r, [y, y, y],
            c=[LONG, RAMP, BRIEF], s=52, zorder=3,
            edgecolors="white", linewidths=0.7,
        )
        ylabels.append((y, f"{short_type(a)}\n{short_type(b)}"))

    groups = []
    i0 = 0
    while i0 < n:
        tag = rows[i0][2]["tag"]
        i1 = i0
        while i1 + 1 < n and rows[i1 + 1][2]["tag"] == tag:
            i1 += 1
        groups.append((tag, n - 1 - i1, n - 1 - i0))
        i0 = i1 + 1

    ax.set_yticks([y for y, _ in ylabels])
    ax.set_yticklabels([lab for _, lab in ylabels], fontsize=7.5, linespacing=1.2)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.65, n - 0.35)
    ax.set_xlabel("Rank of type-distance S  (1 = farthest)")
    ax.spines["left"].set_visible(False)

    # group tags: text only
    y0, y1 = ax.get_ylim()
    for tag, y_lo, y_hi in groups:
        mid = 0.5 * (y_lo + y_hi)
        mid_ax = (mid - y0) / (y1 - y0)
        lo_ax = (y_lo - 0.25 - y0) / (y1 - y0)
        hi_ax = (y_hi + 0.25 - y0) / (y1 - y0)
        ax.plot(
            [1.01, 1.04, 1.04, 1.01], [lo_ax, lo_ax, hi_ax, hi_ax],
            transform=ax.transAxes, color=MUTED, lw=1.1,
            clip_on=False, solid_capstyle="round",
        )
        ax.text(
            1.06, mid_ax, tag, transform=ax.transAxes,
            ha="left", va="center", fontsize=8, color=INK,
            fontweight="normal", clip_on=False,
        )

    # protocol legend above the panel (avoids x-label overlap)
    h1 = ax.scatter([], [], c=LONG, s=28, label="Long")
    h2 = ax.scatter([], [], c=RAMP, s=28, label="Ramp")
    h3 = ax.scatter([], [], c=BRIEF, s=28, label="Brief")
    ax.legend(
        handles=[h1, h2, h3], frameon=False, fontsize=8,
        loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
        handletextpad=0.35, columnspacing=1.2,
    )
    x_letter = letter_fig(ax_a, "A")
    letter_fig(ax, "C", x=x_letter)
    save(fig, "Fig3_compare_backbone")


# ---------------------------------------------------------------------------
# Fig 4 — EXPLAIN
# ---------------------------------------------------------------------------
def fig4_shared(tpairs, mods):
    """Figure 4: RNA vs shared S, rewrite, excitability programs."""
    for r in tpairs:
        for k in ("T_primary", "S_long_pulse", "S_ramp", "S_brief_pulse", "rank_range"):
            r[k] = float(r[k])
    mean_s = np.array([(r["S_long_pulse"] + r["S_ramp"] + r["S_brief_pulse"]) / 3 for r in tpairs])
    T = np.array([r["T_primary"] for r in tpairs])
    rr = np.array([r["rank_range"] for r in tpairs])

    purple = MOD["shared"]
    rose = MOD["rewrite"]
    soft_purple = soft_bar_color(purple, 0.55)
    soft_rose = soft_bar_color(rose, 0.50)
    cmap_shared = hexbin_cmap(purple, "fig4shared")
    cmap_rewrite = hexbin_cmap(rose, "fig4rewrite")

    fig = plt.figure(figsize=(12.6, 4.0))
    gs = gridspec.GridSpec(
        1, 3, wspace=0.34,
        left=0.07, right=0.98, top=0.90, bottom=0.18,
    )

    rho_a = spearman(mean_s, T)
    rho_b = spearman(rr, T)

    # A — RNA vs shared map
    ax = fig.add_subplot(gs[0, 0])
    ax.hexbin(mean_s, T, gridsize=28, cmap=cmap_shared, mincnt=1, linewidths=0)
    ax.set_xlabel("Mean phenotypic S")
    ax.set_ylabel("RNA distance T")
    ax.text(
        0.98, 0.98, f"ρ = {rho_a:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK,
    )
    letter(ax, "A")

    # B — RNA vs rewrite (mirror of A)
    ax = fig.add_subplot(gs[0, 1])
    ax.hexbin(rr, T, gridsize=28, cmap=cmap_rewrite, mincnt=1, linewidths=0)
    ax.set_xlabel("Rank-range (rewrite)")
    ax.set_ylabel("RNA distance T")
    ax.text(
        0.98, 0.98, f"ρ = {rho_b:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK,
    )
    letter(ax, "B")

    # C — excitability modules (no Kv3 headline)
    ax = fig.add_subplot(gs[0, 2])
    excit = [r for r in mods if r["role"] == "excitability"]
    y = np.arange(len(excit))
    ax.barh(
        y + 0.16, [float(r["spearman_D_vs_meanS"]) for r in excit], 0.30,
        color=soft_purple, edgecolor=purple, lw=0.8, label="shared structure",
    )
    ax.barh(
        y - 0.16, [float(r["spearman_D_vs_rank_range"]) for r in excit], 0.30,
        color=soft_rose, edgecolor=rose, lw=0.8, label="regime-selective rewrite",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([r["module"] for r in excit], fontsize=7.5)
    ax.axvline(0, color="#5A6570", lw=0.7)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman of module distance")
    ax.text(
        0.0, 1.04,
        "shared structure  |  regime-selective rewrite",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5, color=MUTED, clip_on=False,
    )
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    letter(ax, "C")
    save(fig, "Fig4_explain_shared")


def main():
    style()
    join = load_join(JOIN)
    pack = {}
    for p, path in SCORE_FILES.items():
        stems, X = load_scores(path)
        pack[p] = {"stems": stems, "X": X, "k": load_k80(VAR_FILES[p])}
    pack["triple"] = sorted(
        set(pack["long_pulse"]["stems"]) & set(pack["ramp"]["stems"]) & set(pack["brief_pulse"]["stems"])
    )
    traces = load_traces()
    print("Fig1…")
    fig1(pack, load_csv(Q1), traces, load_csv(Q1_JAC), load_csv(Q1_SCALE))
    print("Fig2…")
    fig2_2x2(pack, join, load_csv(Q2))
    print("Fig3…")
    fig3_backbone(load_csv(PAIRS), load_csv(CONC))
    print("Fig4…")
    fig4_shared(load_csv(TPAIRS), load_csv(MODS))
    (OUT / "README.md").write_text(
        "# Revision Results figures\n\n"
        "Written to `revision/figures/results`. Submitted originals are unchanged.\n\n"
        "| File | Question | Panels |\n"
        "|------|----------|--------|\n"
        "| `Fig1_describe` | Q1 DESCRIBE | A traces · B pairwise D hexbin + Spearman · C 15-NN Jaccard |\n"
        "| `Fig2_anchor_2x2` | Q2 ANCHOR | A occupancy · B within-Cre · C η² · D kNN enrich |\n"
        "| `Fig3_compare_backbone` | Q3 COMPARE | A conserved S · B tags+% · C grouped trajectories |\n"
        "| `Fig4_explain_shared` | Q4 EXPLAIN | A shared · B rewrite · C excitability programs |\n\n"
        "Fig. 1B/C and Fig. 3A use the same muted pair hues "
        "(Long–Ramp slate #1F3F58, Long–Brief teal #0E4A47, Ramp–Brief umber #542616). "
        "Fig. 3B tag identity is hatch at staggered density; Fig. 3C tags are text.\n"
        "Generated by `revision/scripts/figures_eneuro_results.py`.\n"
    )
    print("done →", OUT)


if __name__ == "__main__":
    main()
