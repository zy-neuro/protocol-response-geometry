#!/usr/bin/env python3
"""within-regime split-half neighborhood reliability.

Split usable long-square sweeps into two disjoint halves, extract IPFX
long-square features on each half, project into the ORIGINAL long-pulse
phenotype space (full-cell z-score + PCA; halves do not enter the fit),
and compare 15-NN Jaccard of half-A vs half-B with cross-regime 15-NN
Jaccard for the same cells.

Writes results/q1_reliability/ only.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (  # noqa: E402
    ORIG_FEATURES,
    ORIG_PHENO,
    OUT_Q1,
    OUT_Q1_REL,
    REPO,
)

SEED = 0
N_SHUFFLE = 100
KNN = 15
MIN_SWEEPS_PER_HALF = 2
MIN_USABLE = 4
VARIANCE_FLOOR = 0.80
PROTOCOL = "long_pulse"
PREFIX = "feat.long_squares."

SKIP_FEATURE_KEYS = {
    "spikes",
    "sweeps",
    "spiking_sweeps",
    "subthreshold_sweeps",
    "subthreshold_membrane_property_sweeps",
    "common_amp_sweeps",
}
LONG_RECORD = {
    "vrest",
    "ri",
    "sag",
    "tau",
    "vm_for_sag",
    "f_i_curve_slope",
    "adaptation",
    "latency",
    "avg_isi",
}
RECORD_SUFFIX = {
    "_long_square": "long_squares",
    "_ramp": "ramps",
    "_short_square": "short_squares",
}


def as_float(value):
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num != num:
        return None
    return num


def flatten(prefix: str, obj, out: dict) -> None:
    if obj is None:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in SKIP_FEATURE_KEYS:
                continue
            flatten(f"{prefix}{k}.", v, out)
        return
    num = as_float(obj)
    if num is not None:
        out[prefix.rstrip(".")] = num


def pack_cell_record(cell_record: dict, rec: dict) -> None:
    for key, value in cell_record.items():
        num = as_float(value)
        if num is None:
            continue
        proto = None
        name = key
        for suffix, protocol in RECORD_SUFFIX.items():
            if key.endswith(suffix):
                proto = protocol
                name = key[: -len(suffix)]
                break
        if proto is None:
            if key in LONG_RECORD:
                proto = "long_squares"
            else:
                continue
        rec[f"feat.{proto}.{name}"] = num


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if val != val else val
    if isinstance(obj, Path):
        return str(obj)
    return obj


def repair_nwb_epochs() -> None:
    from ipfx.sweep import Sweep

    if getattr(Sweep.__init__, "_prg_epoch_repair", False):
        return
    original = Sweep.__init__

    def patched(self, t, v, i, clamp_mode, sampling_rate, sweep_number=None, epochs=None):
        if epochs:
            epochs = {k: val for k, val in epochs.items() if val is not None}
        original(self, t, v, i, clamp_mode, sampling_rate, sweep_number=sweep_number, epochs=epochs)

    patched._prg_epoch_repair = True
    Sweep.__init__ = patched


def _ok_split(a: list[int], b: list[int]) -> bool:
    return len(a) >= MIN_SWEEPS_PER_HALF and len(b) >= MIN_SWEEPS_PER_HALF


def candidate_splits(
    sns: list[int], rng: np.random.Generator, amps: dict[int, float] | None = None
) -> list[tuple[list[int], list[int], str]]:
    sns = [int(s) for s in sns]
    out: list[tuple[list[int], list[int], str]] = []
    even = sorted(s for s in sns if s % 2 == 0)
    odd = sorted(s for s in sns if s % 2 == 1)
    out.append((even, odd, "odd_even_sweep_number"))
    sns_s = sorted(sns)
    out.append((sns_s[0::2], sns_s[1::2], "interleaved_index"))
    if amps:
        ordered = [s for s, _amp, _sn in sorted(((s, amps.get(s, 0.0), s) for s in sns), key=lambda t: (t[1], t[2]))]
        out.append((ordered[0::2], ordered[1::2], "amplitude_interleaved"))
    perm = rng.permutation(len(sns_s))
    mid = max(MIN_SWEEPS_PER_HALF, len(sns_s) // 2)
    if len(sns_s) - mid >= MIN_SWEEPS_PER_HALF:
        a = sorted(int(sns_s[i]) for i in perm[:mid])
        b = sorted(int(sns_s[i]) for i in perm[mid:])
        out.append((a, b, "random_seed0"))
    seen = set()
    uniq = []
    for a, b, rule in out:
        if not _ok_split(a, b):
            continue
        key = (tuple(a), tuple(b), rule)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((a, b, rule))
    return uniq


def _pack_long_square_features(features: dict) -> dict:
    from ipfx.feature_record import build_cell_feature_record

    rec: dict = {}
    cell_features = {"long_squares": features or {}, "ramps": {}, "short_squares": {}}
    try:
        cell_record = build_cell_feature_record(cell_features)
    except Exception:
        cell_record = {}
    if isinstance(cell_record, dict):
        pack_cell_record(cell_record, rec)
    flatten("feat.", {"long_squares": features or {}}, rec)
    return rec


def extract_one_split(nwb_path: str) -> dict:
    os.environ.setdefault("PYNWB_HOME", str(REPO / "data/official/.pynwb"))
    os.environ.setdefault("MPLCONFIGDIR", str(REPO / "data/official/.mpl"))
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("ipfx").setLevel(logging.ERROR)

    path = Path(nwb_path)
    rec = {
        "nwb_path": str(path),
        "file_stem": path.stem,
        "extract_ok": 0,
        "skip_reason": "",
        "n_usable_long_squares": 0,
        "split_rule": "",
        "half_a_sweeps": [],
        "half_b_sweeps": [],
        "half_a_ok": 0,
        "half_b_ok": 0,
        "half_a_fail": "",
        "half_b_fail": "",
        "features_a": {},
        "features_b": {},
    }
    try:
        repair_nwb_epochs()
        from ipfx.dataset.create import create_ephys_data_set
        from ipfx.data_set_features import (
            extract_cell_long_square_features,
            select_subthreshold_min_amplitude,
        )

        data_set = create_ephys_data_set(nwb_file=str(path))
        original_get = data_set.get_sweep_numbers
        allowed_holder: dict = {"set": None}

        def filtered(*args, **kwargs):
            keep = []
            allowed = allowed_holder["set"]
            for sn in original_get(*args, **kwargs):
                if allowed is not None and int(sn) not in allowed:
                    continue
                sweep = data_set.sweep(int(sn))
                if sweep.epochs.get("experiment") and sweep.epochs.get("stim"):
                    keep.append(sn)
            return keep

        data_set.get_sweep_numbers = filtered
        usable = [int(s) for s in data_set.get_sweep_numbers(
            data_set.ontology.long_square_names, clamp_mode=data_set.CURRENT_CLAMP
        )]
        rec["n_usable_long_squares"] = len(usable)
        if len(usable) < MIN_USABLE:
            rec["skip_reason"] = f"n_usable={len(usable)}<{MIN_USABLE}"
            return rec

        amp_map: dict[int, float] = {}
        try:
            table = data_set.filtered_sweep_table(
                clamp_mode=data_set.CURRENT_CLAMP,
                stimuli=data_set.ontology.long_square_names,
            )
            if table is not None and "sweep_number" in table and "stimulus_amplitude" in table:
                for sn, amp in zip(table["sweep_number"], table["stimulus_amplitude"]):
                    try:
                        amp_map[int(sn)] = float(amp)
                    except (TypeError, ValueError):
                        continue
        except Exception:
            amp_map = {}

        rng = np.random.default_rng(SEED)
        splits = candidate_splits(usable, rng, amp_map or None)
        if not splits:
            rec["skip_reason"] = "could_not_split"
            return rec

        clsq = data_set.filtered_sweep_table(
            clamp_mode=data_set.CURRENT_CLAMP,
            stimuli=data_set.ontology.coarse_long_square_names,
        )
        if clsq is not None and len(clsq) > 0 and "stimulus_amplitude" in clsq:
            subthresh_min_amp, _delta = select_subthreshold_min_amplitude(clsq["stimulus_amplitude"])
        else:
            subthresh_min_amp = -100

        def run_half(sweeps: list[int]) -> tuple[int, str, dict]:
            allowed_holder["set"] = set(int(s) for s in sweeps)
            try:
                features, state = extract_cell_long_square_features(
                    data_set, subthresh_min_amp=subthresh_min_amp
                )
            except Exception as exc:
                return 0, f"{type(exc).__name__}: {exc}", {}
            failed = bool((state or {}).get("failed_fx"))
            if failed or not features:
                return 0, str((state or {}).get("fail_fx_message") or "failed_fx"), {}
            return 1, "", _pack_long_square_features(features)

        last_fail = ""
        for half_a, half_b, rule in splits:
            ok_a, fail_a, feat_a = run_half(half_a)
            ok_b, fail_b, feat_b = run_half(half_b)
            rec["split_rule"] = rule
            rec["half_a_sweeps"] = half_a
            rec["half_b_sweeps"] = half_b
            rec["half_a_ok"] = ok_a
            rec["half_b_ok"] = ok_b
            rec["half_a_fail"] = fail_a
            rec["half_b_fail"] = fail_b
            rec["features_a"] = feat_a
            rec["features_b"] = feat_b
            if ok_a and ok_b:
                rec["extract_ok"] = 1
                rec["skip_reason"] = ""
                return rec
            last_fail = fail_a or fail_b or "half_failed"
        rec["skip_reason"] = last_fail or "half_failed"
        return rec
    except Exception as exc:
        rec["skip_reason"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()
        return rec


def _init_worker() -> None:
    os.environ.setdefault("PYNWB_HOME", str(REPO / "data/official/.pynwb"))
    os.environ.setdefault("MPLCONFIGDIR", str(REPO / "data/official/.mpl"))
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    logging.getLogger().setLevel(logging.ERROR)


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


def chance_jaccard(idx_a: np.ndarray, idx_b: np.ndarray, rng: np.random.Generator, n_shuffle: int) -> tuple[float, float]:
    n = idx_a.shape[0]
    null_means = np.empty(n_shuffle, dtype=float)
    for t in range(n_shuffle):
        perm = rng.permutation(n)
        null_means[t] = float(np.mean(jaccard_rows(idx_a, idx_b[perm])))
    return float(np.mean(null_means)), float(np.std(null_means))


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


def fit_full_cell_pca(feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Fit z-score + PCA on submitted cell-level long-square features. No half labels."""
    src = ORIG_FEATURES / "ipfx_cell_features.csv"
    rows = [r for r in load_csv(src) if r.get("extract_ok") == "1"]
    mat = np.full((len(rows), len(feature_cols)), np.nan, dtype=float)
    stems = []
    for i, row in enumerate(rows):
        stems.append(row.get("file_stem") or "")
        for j, c in enumerate(feature_cols):
            v = row.get(c, "")
            if v in ("", None):
                continue
            try:
                mat[i, j] = float(v)
            except ValueError:
                continue
    present = np.isfinite(mat).any(axis=1)
    mat = mat[present]
    stems = [s for s, ok in zip(stems, present) if ok]
    mu = np.nanmean(mat, axis=0)
    xf = np.where(np.isfinite(mat), mat, mu)
    xc = xf - xf.mean(axis=0, keepdims=True)
    std = xc.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    z = xc / std
    _u, _s, vt = np.linalg.svd(z, full_matrices=False)
    return mu, std.ravel(), vt, stems


def project_halves(rows: list[dict], feature_cols: list[str], mu: np.ndarray, std: np.ndarray, vt: np.ndarray, k80: int):
    keep = [r for r in rows if int(r.get("extract_ok") or 0) == 1]
    stems = [r["file_stem"] for r in keep]
    xa = np.full((len(keep), len(feature_cols)), np.nan)
    xb = np.full((len(keep), len(feature_cols)), np.nan)
    for i, rec in enumerate(keep):
        fa = rec.get("features_a") or {}
        fb = rec.get("features_b") or {}
        for j, c in enumerate(feature_cols):
            xa[i, j] = np.nan if fa.get(c) is None else float(fa[c])
            xb[i, j] = np.nan if fb.get(c) is None else float(fb[c])
    ok = np.isfinite(xa).any(axis=1) & np.isfinite(xb).any(axis=1)
    stems = [s for s, flag in zip(stems, ok) if flag]
    xa, xb = xa[ok], xb[ok]

    def proj(x):
        xf = np.where(np.isfinite(x), x, mu)
        z = (xf - mu) / std
        return z @ vt.T[:, :k80]

    return stems, proj(xa), proj(xb)


def gather_parts(part_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(part_dir.glob("*.json")):
        rec = json.loads(path.read_text())
        rows.append(rec)
    return rows


def write_feature_csv(rows: list[dict], path: Path, feature_cols: list[str]) -> None:
    out = []
    for rec in rows:
        base = {
            "file_stem": rec.get("file_stem"),
            "extract_ok": rec.get("extract_ok"),
            "skip_reason": rec.get("skip_reason", ""),
            "n_usable_long_squares": rec.get("n_usable_long_squares"),
            "split_rule": rec.get("split_rule", ""),
            "half_a_ok": rec.get("half_a_ok"),
            "half_b_ok": rec.get("half_b_ok"),
            "half_a_sweeps": " ".join(str(s) for s in rec.get("half_a_sweeps") or []),
            "half_b_sweeps": " ".join(str(s) for s in rec.get("half_b_sweeps") or []),
            "half_a_fail": rec.get("half_a_fail", ""),
            "half_b_fail": rec.get("half_b_fail", ""),
        }
        fa = rec.get("features_a") or {}
        fb = rec.get("features_b") or {}
        for c in feature_cols:
            base[f"a.{c}"] = fa.get(c, "")
            base[f"b.{c}"] = fb.get(c, "")
        out.append(base)
    fields = list(out[0].keys()) if out else []
    write_csv(path, out, fields)


def run_extract(nwb_paths: list[Path], part_dir: Path, workers: int) -> None:
    part_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in nwb_paths if not (part_dir / f"{p.stem}.json").exists()]
    print(f"Extract split-half: {len(todo)} remaining / {len(nwb_paths)} NWB", flush=True)
    if not todo:
        return
    workers = max(1, int(workers))
    if workers == 1:
        for i, path in enumerate(todo, 1):
            rec = extract_one_split(str(path))
            (part_dir / f"{path.stem}.json").write_text(json.dumps(jsonable(rec)) + "\n")
            if i % 25 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] {path.stem} ok={rec.get('extract_ok')} {rec.get('skip_reason','')}", flush=True)
        return
    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        futs = {ex.submit(extract_one_split, str(p)): p for p in todo}
        for fut in as_completed(futs):
            path = futs[fut]
            done += 1
            try:
                rec = fut.result()
            except Exception as exc:
                rec = {"file_stem": path.stem, "nwb_path": str(path), "extract_ok": 0, "skip_reason": f"worker: {exc}"}
            (part_dir / f"{path.stem}.json").write_text(json.dumps(jsonable(rec)) + "\n")
            if done % 25 == 0 or done == len(todo):
                print(
                    f"  [{done}/{len(todo)}] {path.stem} ok={rec.get('extract_ok')} {rec.get('skip_reason','')}",
                    flush=True,
                )


def analyze(rows: list[dict]) -> dict:
    summary_path = ORIG_PHENO / "phenotype_summary.json"
    feature_cols = list(json.loads(summary_path.read_text())[PROTOCOL]["features"])
    mu, std, vt, _full_stems = fit_full_cell_pca(feature_cols)
    k80 = load_k80(ORIG_PHENO / f"{PROTOCOL}_pca_variance.csv", vt.shape[0])
    stems, sa, sb = project_halves(rows, feature_cols, mu, std, vt, k80)
    n = len(stems)
    print(f"Projected split-half cells: n={n} k80={k80}", flush=True)
    if n < KNN + 2:
        payload = {
            "n_nwb_attempted": len(rows),
            "n_eligible_ge4_sweeps": sum(1 for r in rows if int(r.get("n_usable_long_squares") or 0) >= MIN_USABLE),
            "n_both_halves_ok": sum(1 for r in rows if int(r.get("extract_ok") or 0) == 1),
            "n_projected": n,
            "k": KNN,
            "k80_long_pulse": k80,
            "within_jaccard_mean": float("nan"),
            "within_jaccard_median": float("nan"),
            "within_chance_mean": float("nan"),
            "within_adjusted": float("nan"),
            "cross_samepool_means": {},
            "cross_samepool_mean_of_pairs": float("nan"),
            "landing": f"too few split-half cells for 15-NN (n={n})",
            "projection": "full-cell long-pulse z-score + PCA; halves projected, not used in the fit",
            "split": "odd/even long-square sweep number (fallback interleaved index, random seed=0)",
            "nwb_used": True,
        }
        (OUT_Q1_REL / "split_half_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
        return payload

    da = pairwise_euclidean(sa)
    db = pairwise_euclidean(sb)
    knn_a = knn_indices(da, KNN)
    knn_b = knn_indices(db, KNN)
    jac_within = jaccard_rows(knn_a, knn_b)
    rng = np.random.default_rng(SEED)
    chance_mean, chance_sd = chance_jaccard(knn_a, knn_b, rng, N_SHUFFLE)

    # Cross-regime Jaccard already computed on the full shared-cell pools.
    jac_path = OUT_Q1 / "q1_knn_jaccard_k15.csv"
    by_pair: dict[str, dict[str, float]] = {}
    for r in load_csv(jac_path):
        by_pair.setdefault(r["pair"], {})[r["file_stem"]] = float(r["jaccard_k15"])

    pair_names = [
        "long_pulse__ramp",
        "long_pulse__brief_pulse",
        "ramp__brief_pulse",
    ]

    # Fair recompute: kNN among the same split-half cells that exist in both regimes.
    proto_scores = {}
    proto_k = {}
    for p in ("long_pulse", "ramp", "brief_pulse"):
        s, mat = load_scores(ORIG_PHENO / f"{p}_pca_scores.csv")
        proto_scores[p] = {stem: mat[i] for i, stem in enumerate(s)}
        proto_k[p] = load_k80(ORIG_PHENO / f"{p}_pca_variance.csv", mat.shape[1])

    def recompute_pair(pa: str, pb: str, cell_list: list[str]) -> tuple[np.ndarray, list[str]]:
        shared = [s for s in cell_list if s in proto_scores[pa] and s in proto_scores[pb]]
        xa = np.vstack([proto_scores[pa][s][: proto_k[pa]] for s in shared])
        xb = np.vstack([proto_scores[pb][s][: proto_k[pb]] for s in shared])
        ia = knn_indices(pairwise_euclidean(xa), KNN)
        ib = knn_indices(pairwise_euclidean(xb), KNN)
        return jaccard_rows(ia, ib), shared

    cell_rows = []
    for i, stem in enumerate(stems):
        row = {"file_stem": stem, "jaccard_within_k15": float(jac_within[i])}
        for pair in pair_names:
            row[f"jaccard_{pair}_fullpool"] = by_pair.get(pair, {}).get(stem, "")
        cell_rows.append(row)

    # attach recomputed same-pool Jaccard
    pair_recomp = {}
    for pair, (pa, pb) in zip(
        pair_names,
        (("long_pulse", "ramp"), ("long_pulse", "brief_pulse"), ("ramp", "brief_pulse")),
    ):
        jac, shared = recompute_pair(pa, pb, stems)
        pair_recomp[pair] = {s: float(v) for s, v in zip(shared, jac)}
        ch_mean, ch_sd = chance_jaccard(
            knn_indices(pairwise_euclidean(np.vstack([proto_scores[pa][s][: proto_k[pa]] for s in shared])), KNN),
            knn_indices(pairwise_euclidean(np.vstack([proto_scores[pb][s][: proto_k[pb]] for s in shared])), KNN),
            np.random.default_rng(SEED),
            N_SHUFFLE,
        )
        pair_recomp[pair + "__chance"] = (ch_mean, ch_sd, len(shared))

    for row in cell_rows:
        stem = row["file_stem"]
        crosses = []
        for pair in pair_names:
            val = pair_recomp[pair].get(stem)
            row[f"jaccard_{pair}_samepool"] = "" if val is None else val
            if val is not None:
                crosses.append(val)
            fullv = row[f"jaccard_{pair}_fullpool"]
            if fullv != "":
                row[f"jaccard_{pair}_fullpool"] = float(fullv)
        row["jaccard_cross_mean_samepool"] = float(np.mean(crosses)) if crosses else ""

    write_csv(OUT_Q1_REL / "split_half_jaccard_cells.csv", cell_rows)

    within_vals = [float(r["jaccard_within_k15"]) for r in cell_rows]
    summary_rows = [
        {
            "contrast": "within_long_square_split_half",
            "pool": "split_half_cells",
            "n_cells": int(n),
            "observed_jaccard_mean": float(np.mean(within_vals)),
            "observed_jaccard_median": float(np.median(within_vals)),
            "chance_jaccard_mean": chance_mean,
            "chance_jaccard_sd": chance_sd,
            "adjusted_jaccard": float(np.mean(within_vals) - chance_mean),
            "n_shuffle": N_SHUFFLE,
            "k": KNN,
        }
    ]
    for pair in pair_names:
        full_vals = [float(r[f"jaccard_{pair}_fullpool"]) for r in cell_rows if r[f"jaccard_{pair}_fullpool"] not in ("", None)]
        same_vals = [float(r[f"jaccard_{pair}_samepool"]) for r in cell_rows if r[f"jaccard_{pair}_samepool"] not in ("", None)]
        ch_mean, ch_sd, n_shared = pair_recomp[pair + "__chance"]
        summary_rows.append(
            {
                "contrast": pair,
                "pool": "same_split_half_cells",
                "n_cells": int(len(same_vals)),
                "observed_jaccard_mean": float(np.mean(same_vals)) if same_vals else float("nan"),
                "observed_jaccard_median": float(np.median(same_vals)) if same_vals else float("nan"),
                "chance_jaccard_mean": ch_mean,
                "chance_jaccard_sd": ch_sd,
                "adjusted_jaccard": float(np.mean(same_vals) - ch_mean) if same_vals else float("nan"),
                "n_shuffle": N_SHUFFLE,
                "k": KNN,
            }
        )
        summary_rows.append(
            {
                "contrast": pair,
                "pool": "original_full_shared_pool_subset",
                "n_cells": int(len(full_vals)),
                "observed_jaccard_mean": float(np.mean(full_vals)) if full_vals else float("nan"),
                "observed_jaccard_median": float(np.median(full_vals)) if full_vals else float("nan"),
                "chance_jaccard_mean": "",
                "chance_jaccard_sd": "",
                "adjusted_jaccard": "",
                "n_shuffle": "",
                "k": KNN,
            }
        )

    write_csv(OUT_Q1_REL / "split_half_summary.csv", summary_rows)

    n_attempted = len(rows)
    n_ok = sum(1 for r in rows if int(r.get("extract_ok") or 0) == 1)
    n_eligible = sum(1 for r in rows if int(r.get("n_usable_long_squares") or 0) >= MIN_USABLE)
    within_mean = float(np.mean(within_vals))
    cross_same = [
        float(r["observed_jaccard_mean"])
        for r in summary_rows
        if r["pool"] == "same_split_half_cells" and r["contrast"] != "within_long_square_split_half"
    ]
    cross_mean = float(np.mean(cross_same)) if cross_same else float("nan")
    if within_mean > 3.0 * cross_mean:
        landing = (
            "within-regime trial reliability >> cross-regime overlap; "
            "stimulus-dependent reorganization is supported"
        )
    elif within_mean > 1.5 * cross_mean:
        landing = (
            "within-regime trial reliability is higher than cross-regime overlap, "
            "supporting stimulus-dependent reorganization with residual trial noise"
        )
    else:
        landing = (
            "within-regime trial reliability is similar to cross-regime overlap; "
            "cross-regime Jaccard should be interpreted more cautiously"
        )

    payload = {
        "n_nwb_attempted": n_attempted,
        "n_eligible_ge4_sweeps": n_eligible,
        "n_both_halves_ok": n_ok,
        "n_projected": n,
        "k": KNN,
        "k80_long_pulse": k80,
        "within_jaccard_mean": within_mean,
        "within_jaccard_median": float(np.median(within_vals)),
        "within_chance_mean": chance_mean,
        "within_adjusted": within_mean - chance_mean,
        "cross_samepool_means": {
            r["contrast"]: r["observed_jaccard_mean"]
            for r in summary_rows
            if r["pool"] == "same_split_half_cells"
        },
        "cross_samepool_mean_of_pairs": cross_mean,
        "landing": landing,
        "projection": "full-cell long-pulse z-score + PCA; halves projected, not used in the fit",
            "split": "odd/even long-square sweep number (fallback interleaved index, random seed=0)",
        "nwb_used": True,
    }
    (OUT_Q1_REL / "split_half_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def write_landing(rows: list[dict], payload: dict) -> None:
    skip = {}
    for r in rows:
        key = r.get("skip_reason") or ("ok" if int(r.get("extract_ok") or 0) else "other")
        skip[key] = skip.get(key, 0) + 1
    lines = [
        "# Q1 split-half trial reliability",
        "",
        "Within-regime control: two independent long-square phenotype estimates per cell, "
        "projected into the original long-pulse k₈₀ space, then 15-NN Jaccard of half-A vs half-B.",
        "",
        f"**Landing:** {payload['landing']}",
        "",
        "## Numbers",
        "",
        f"- NWB files attempted: {payload['n_nwb_attempted']}",
        f"- Eligible (≥4 usable long-square sweeps): {payload['n_eligible_ge4_sweeps']}",
        f"- Both halves extracted: {payload['n_both_halves_ok']}",
        f"- Projected for kNN: {payload['n_projected']}",
        f"- Within-regime 15-NN Jaccard mean: {payload['within_jaccard_mean']:.4f} "
        f"(chance {payload['within_chance_mean']:.4f}; adjusted {payload['within_adjusted']:.4f})",
        f"- Cross-regime 15-NN Jaccard (same cells, kNN recomputed in that pool), mean of three pairs: "
        f"{payload['cross_samepool_mean_of_pairs']:.4f}",
        "",
        "### Cross-regime (same pool)",
        "",
    ]
    for k, v in payload["cross_samepool_means"].items():
        lines.append(f"- {k}: {float(v):.4f}")
    lines += [
        "",
        "## Design",
        "",
        "- Source: official DANDI 000020 NWBs (`data/official/dandi_000020/`).",
        "- Epoch-None repair copied from `scripts/pipeline/01_extract_features.py`.",
        "- Split: odd/even long-square sweep number (≥2 sweeps per half); fallback interleaved index, then random seed=0.",
        "- PCA fit uses submitted cell-level long-square features only (no half labels).",
        "- Chance: 100 index shuffles of cell pairing (seed 0).",
        "",
        "## Skip reasons",
        "",
    ]
    for k, n in sorted(skip.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {k}: {n}")
    (OUT_Q1_REL / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="first N NWB files (smoke)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()

    OUT_Q1_REL.mkdir(parents=True, exist_ok=True)
    part_dir = OUT_Q1_REL / "parts"
    dandi = REPO / "data/official/dandi_000020"
    nwbs = sorted(dandi.rglob("*_icephys.nwb"))
    if not nwbs:
        raise SystemExit(f"No *_icephys.nwb under {dandi}")
    if args.limit:
        nwbs = nwbs[: args.limit]

    if not args.analyze_only and not args.skip_extract:
        run_extract(nwbs, part_dir, args.workers)

    rows = gather_parts(part_dir)
    if args.limit and not args.analyze_only:
        keep = {p.stem for p in nwbs}
        rows = [r for r in rows if r.get("file_stem") in keep]
    print(f"Gathered {len(rows)} part files", flush=True)
    n_ok = sum(1 for r in rows if int(r.get("extract_ok") or 0) == 1)
    print(f"  extract_ok={n_ok}/{len(rows)}", flush=True)
    if args.extract_only:
        inv = [
            {
                "file_stem": r.get("file_stem"),
                "n_usable_long_squares": r.get("n_usable_long_squares"),
                "split_rule": r.get("split_rule"),
                "extract_ok": r.get("extract_ok"),
                "skip_reason": r.get("skip_reason", ""),
                "half_a_ok": r.get("half_a_ok"),
                "half_b_ok": r.get("half_b_ok"),
                "half_a_n": len(r.get("half_a_sweeps") or []),
                "half_b_n": len(r.get("half_b_sweeps") or []),
            }
            for r in rows
        ]
        write_csv(OUT_Q1_REL / "split_half_inventory.csv", inv)
        print("extract-only: skipping projection/Jaccard")
        return
    feature_cols = list(json.loads((ORIG_PHENO / "phenotype_summary.json").read_text())[PROTOCOL]["features"])
    write_feature_csv(rows, OUT_Q1_REL / "split_half_features.csv", feature_cols)
    inv = [
        {
            "file_stem": r.get("file_stem"),
            "n_usable_long_squares": r.get("n_usable_long_squares"),
            "split_rule": r.get("split_rule"),
            "extract_ok": r.get("extract_ok"),
            "skip_reason": r.get("skip_reason", ""),
            "half_a_ok": r.get("half_a_ok"),
            "half_b_ok": r.get("half_b_ok"),
            "half_a_n": len(r.get("half_a_sweeps") or []),
            "half_b_n": len(r.get("half_b_sweeps") or []),
        }
        for r in rows
    ]
    write_csv(OUT_Q1_REL / "split_half_inventory.csv", inv)
    payload = analyze(rows)
    write_landing(rows, payload)
    print(json.dumps({k: payload[k] for k in ("n_projected", "within_jaccard_mean", "cross_samepool_mean_of_pairs", "landing")}, indent=2))
    print(f"Wrote {OUT_Q1_REL}")


if __name__ == "__main__":
    main()
