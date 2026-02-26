#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SEM edge-based segmentation toolkit (macOS-friendly)


# Example terminal usage
# cd "/Users/joohopark/Library/CloudStorage/Dropbox/Jooho/Clustering"
# source .venv/bin/activate
# ls *.tif
#
# Single run:
# python sem_segmentation_mac.py run --infile "./#1 Snapshot (Map + Point).tif" --k 4 --win 25
#
# Sweep run:
# python sem_segmentation_mac.py sweep --input "./#1 Snapshot (Map + Point).tif" --ks 2 3 4 5 --wins 5 10 20 40
#
# Sweep diver tif:
# python sem_segmentation_mac.py sweep --input "./*.tif" --ks 2 3 4 5 --wins 5 10 20 40

Key behavior for easy use:
- Put this .py file and a .tif/.tiff with the sam e stem in the same folder.
  Example: sem_segmentation_mac.py + sem_segmentation_mac.tif
- Run without arguments:
    python3 sem_segmentation_mac.py
  It will auto-run on the matching TIFF and save outputs to <image_stem>_out.

CLI modes are still available:
- run: process one image
- sweep: process multiple images and parameter combinations
"""

import argparse
import csv
import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from scipy.ndimage import distance_transform_edt
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score, silhouette_score
    from sklearn.preprocessing import StandardScaler
    from skimage import color, filters, io, morphology, transform, util
    from skimage.morphology import skeletonize
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: "
        f"{exc.name}\nInstall required packages:\n"
        "python3 -m pip install numpy scipy scikit-image scikit-learn matplotlib"
    ) from exc


SEM_LABEL_COLORS = np.array(
    [
        [0.306, 0.475, 0.655, 1.0],  # blue
        [0.949, 0.557, 0.169, 1.0],  # orange
        [0.882, 0.341, 0.349, 1.0],  # coral
        [0.463, 0.718, 0.698, 1.0],  # teal
        [0.349, 0.631, 0.310, 1.0],  # green
        [0.929, 0.788, 0.282, 1.0],  # mustard
        [0.690, 0.478, 0.631, 1.0],  # mauve
        [1.000, 0.616, 0.655, 1.0],  # pink
        [0.612, 0.459, 0.373, 1.0],  # brown
        [0.729, 0.690, 0.675, 1.0],  # warm gray
    ],
    dtype=np.float32,
)


# -----------------------------
# I/O
# -----------------------------
def load_image(path: str) -> np.ndarray:
    """
    Load image (tif/png/jpg) as float32 grayscale in [0,1].
    Robust to RGB / RGBA / grayscale.
    """
    img = io.imread(path)

    if img.ndim == 3:
        c = img.shape[-1]
        if c == 4:  # RGBA
            img = img[..., :3]
            img = color.rgb2gray(img)
        elif c == 3:  # RGB
            img = color.rgb2gray(img)
        else:
            img = img[..., 0]

    return util.img_as_float32(img)


def crop_bottom_fraction(img: np.ndarray, frac: float) -> np.ndarray:
    """Remove bottom fraction of image. frac=0.0625 removes bottom 6.25%."""
    if not (0.0 <= frac < 1.0):
        raise ValueError(f"crop_bottom_frac must be in [0,1). Got {frac}")
    h = img.shape[0]
    cut = int(np.floor(h * (1.0 - frac)))
    if cut <= 0:
        raise ValueError(f"Cropping removes entire image. h={h}, frac={frac} -> cut={cut}")
    return img[:cut, :]


def maybe_resize_min(img: np.ndarray, min_dim: int) -> np.ndarray:
    """If max(H,W) > min_dim, downscale so that max(H,W)=min_dim."""
    if min_dim is None or min_dim <= 0:
        return img
    h, w = img.shape
    m = max(h, w)
    if m <= min_dim:
        return img
    scale = min_dim / float(m)
    out = transform.resize(
        img,
        (int(round(h * scale)), int(round(w * scale))),
        order=1,
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)
    return out


# -----------------------------
# Feature map
# -----------------------------
def edge_map(img: np.ndarray, method: str, win: int) -> np.ndarray:
    """
    Compute edge/sharpness map:
    - optional Gaussian smoothing (win>1 => sigma ~ (win-1)/6)
    - sobel or abs(laplacian)
    Output normalized to [0,1].
    """
    if win is None:
        win = 1

    if win > 1:
        sigma = max((win - 1) / 6.0, 0.0)
        work = filters.gaussian(img, sigma=sigma, preserve_range=True)
    else:
        work = img

    if method == "sobel":
        emap = filters.sobel(work)
    elif method == "laplacian":
        emap = np.abs(filters.laplace(work))
    else:
        raise ValueError(f"Unknown method: {method}")

    e_min, e_max = float(np.min(emap)), float(np.max(emap))
    if e_max > e_min:
        emap = (emap - e_min) / (e_max - e_min)
    else:
        emap = np.zeros_like(emap, dtype=np.float32)

    return emap.astype(np.float32)


# -----------------------------
# Segmentation
# -----------------------------
def seg_otsu(emap: np.ndarray) -> np.ndarray:
    thr = filters.threshold_otsu(emap)
    return emap >= thr


def relabel_by_edge_strength(labels: np.ndarray, emap: np.ndarray) -> np.ndarray:
    """Stable semantics: label 0 has lowest mean edge, label k-1 highest."""
    labs = np.unique(labels)
    pairs = []
    for l in labs:
        pairs.append((int(l), float(np.mean(emap[labels == l]))))
    pairs.sort(key=lambda x: x[1])
    mapping = {old: new for new, (old, _) in enumerate(pairs)}

    out = np.zeros_like(labels, dtype=np.int32)
    for old, new in mapping.items():
        out[labels == old] = new
    return out


def _k_selection_indices(
    Xs: np.ndarray,
    labels_1d: np.ndarray,
    k: int,
    seed: int,
    max_eval: int = 20000,
) -> dict:
    """Compute silhouette and CH scores on full or subsampled data."""
    n = int(Xs.shape[0])
    out = {
        "k_eval_n": 0,
        "kmeans_silhouette": float("nan"),
        "kmeans_calinski_harabasz": float("nan"),
    }

    if k < 2 or n < 3:
        return out

    if np.unique(labels_1d).size < 2:
        return out

    if max_eval is not None and max_eval > 0 and n > max_eval:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(n, size=int(max_eval), replace=False)
        X_eval = Xs[idx]
        y_eval = labels_1d[idx]
    else:
        X_eval = Xs
        y_eval = labels_1d

    out["k_eval_n"] = int(X_eval.shape[0])

    if np.unique(y_eval).size < 2:
        return out

    try:
        out["kmeans_silhouette"] = float(silhouette_score(X_eval, y_eval, metric="euclidean"))
    except Exception:
        out["kmeans_silhouette"] = float("nan")

    try:
        out["kmeans_calinski_harabasz"] = float(calinski_harabasz_score(X_eval, y_eval))
    except Exception:
        out["kmeans_calinski_harabasz"] = float("nan")

    return out


def seg_kmeans(
    img: np.ndarray,
    emap: np.ndarray,
    k: int,
    seed: int = 0,
    features: str = "edge+intensity",
    w_edge: float = 1.0,
    w_int: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """
    k-means on selected feature map(s).
    Returns relabeled labels and k-selection metrics.
    """
    if k is None or k < 2:
        raise ValueError("k must be >= 2 for kmeans mode.")
    if features not in ("edge", "edge+intensity"):
        raise ValueError(
            f"features must be 'edge' or 'edge+intensity'. Got {features}"
        )
    if w_edge is None or float(w_edge) <= 0:
        raise ValueError(f"w_edge must be > 0. Got {w_edge}")
    if w_int is None or float(w_int) <= 0:
        raise ValueError(f"w_int must be > 0. Got {w_int}")

    cols = [emap.reshape(-1, 1).astype(np.float32)]
    if features == "edge+intensity":
        cols.append(img.reshape(-1, 1).astype(np.float32))
    X = np.concatenate(cols, axis=1)

    Xs = StandardScaler().fit_transform(X).astype(np.float32)
    Xs[:, 0] *= float(w_edge)
    if features == "edge+intensity":
        Xs[:, 1] *= float(w_int)

    # n_init=10 is compatible with wider sklearn versions.
    km = KMeans(n_clusters=k, n_init=10, random_state=seed, max_iter=500)
    labels_1d = km.fit_predict(Xs).astype(np.int32)

    info = {
        "kmeans_inertia": float(km.inertia_),
        "kmeans_features": str(features),
        "kmeans_w_edge": float(w_edge),
        "kmeans_w_int": float(w_int),
    }
    info.update(_k_selection_indices(Xs, labels_1d, k=k, seed=seed, max_eval=20000))

    labels = labels_1d.reshape(emap.shape).astype(np.int32)
    labels = relabel_by_edge_strength(labels, emap)
    return labels, info


def postprocess_binary(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size is None or min_size <= 0:
        return mask.astype(bool)
    return _remove_small_objects_compat(mask.astype(bool), min_size=int(min_size))


def _remove_small_objects_compat(mask: np.ndarray, min_size: int) -> np.ndarray:
    """
    Compat wrapper for skimage>=0.26 and older versions.
    Old API: min_size removes objects with area < min_size.
    New API: max_size removes objects with area <= max_size.
    Equivalent mapping is max_size = min_size - 1.
    """
    if min_size <= 0:
        return mask.astype(bool)
    try:
        return morphology.remove_small_objects(mask.astype(bool), max_size=max(int(min_size) - 1, 0))
    except TypeError:
        return morphology.remove_small_objects(mask.astype(bool), min_size=int(min_size))


def postprocess_kmeans_labels(labels: np.ndarray, min_size: int) -> tuple[np.ndarray, int]:
    """
    Remove tiny islands per label and reassign dropped pixels by nearest valid label.
    Returns (cleaned_labels, reassigned_pixel_count).
    """
    if min_size is None or int(min_size) <= 0:
        return labels.astype(np.int32), 0

    labels = labels.astype(np.int32)
    out = labels.copy()
    removed_mask = np.zeros(labels.shape, dtype=bool)

    for lab in np.unique(labels):
        mask = labels == int(lab)
        keep = _remove_small_objects_compat(mask, min_size=int(min_size))
        dropped = mask & ~keep
        if np.any(dropped):
            out[dropped] = -1
            removed_mask |= dropped

    n_removed = int(np.sum(removed_mask))
    if n_removed == 0:
        return out, 0

    valid = out >= 0
    if not np.any(valid):
        return labels, 0

    _, nn_idx = distance_transform_edt(~valid, return_indices=True)
    out[~valid] = out[nn_idx[0][~valid], nn_idx[1][~valid]]
    return out.astype(np.int32), n_removed


def _mask_thinness(mask: np.ndarray) -> float:
    """Approximate line-likeness as perimeter pixels / area pixels."""
    area = int(np.sum(mask))
    if area <= 0:
        return float("nan")
    # `binary_erosion` is deprecated in skimage 0.26; use `erosion`.
    eroded = morphology.erosion(mask.astype(bool), footprint=morphology.diamond(1))
    perimeter_px = int(np.sum(mask & ~eroded))
    return float(perimeter_px / float(area))


def infer_boundary_label(
    labels: np.ndarray,
    emap: np.ndarray,
    class_stats: list[dict],
    max_area_frac: float = 0.12,
    min_thinness: float = 0.35,
    min_edge_quantile: float = 0.75,
) -> tuple[Optional[int], dict]:
    """
    Infer boundary label as a class that is:
    - small area fraction
    - line-like (high thinness)
    - in upper edge-mean quantile
    """
    if not class_stats:
        return None, {
            "method": "thin_high_edge_small_area",
            "max_area_frac": float(max_area_frac),
            "min_thinness": float(min_thinness),
            "min_edge_quantile": float(min_edge_quantile),
            "per_label": [],
        }

    edge_means = np.array([float(r["edge_mean"]) for r in class_stats], dtype=np.float32)
    edge_thr = float(np.quantile(edge_means, float(min_edge_quantile)))

    rows = []
    for row in class_stats:
        lab = int(row["label"])
        mask = labels == lab
        thinness = _mask_thinness(mask)
        score = float(row["edge_mean"]) * (1.0 + (0.0 if np.isnan(thinness) else thinness))
        is_candidate = (
            float(row["frac"]) <= float(max_area_frac)
            and (not np.isnan(thinness))
            and thinness >= float(min_thinness)
            and float(row["edge_mean"]) >= edge_thr
        )
        rows.append(
            {
                "label": lab,
                "frac": float(row["frac"]),
                "edge_mean": float(row["edge_mean"]),
                "thinness": float(thinness),
                "edge_threshold": edge_thr,
                "score": float(score),
                "is_candidate": bool(is_candidate),
            }
        )

    candidates = [r for r in rows if r["is_candidate"]]
    if not candidates:
        return None, {
            "method": "thin_high_edge_small_area",
            "max_area_frac": float(max_area_frac),
            "min_thinness": float(min_thinness),
            "min_edge_quantile": float(min_edge_quantile),
            "per_label": rows,
        }

    best = max(candidates, key=lambda r: float(r["score"]))
    return int(best["label"]), {
        "method": "thin_high_edge_small_area",
        "max_area_frac": float(max_area_frac),
        "min_thinness": float(min_thinness),
        "min_edge_quantile": float(min_edge_quantile),
        "per_label": rows,
    }


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(
    img: np.ndarray,
    emap: np.ndarray,
    seg,
    mode: str,
    exclude_labels: Optional[Sequence[int]] = None,
) -> dict:
    out = {
        "mode": mode,
        "H": int(img.shape[0]),
        "W": int(img.shape[1]),
        "n_pixels": int(img.size),
    }

    if mode == "otsu":
        mask = seg.astype(bool)
        out["foreground_frac"] = float(np.mean(mask))
        out["edge_mean_fg"] = float(np.mean(emap[mask])) if np.any(mask) else float("nan")
        out["edge_mean_bg"] = float(np.mean(emap[~mask])) if np.any(~mask) else float("nan")
        out["int_mean_fg"] = float(np.mean(img[mask])) if np.any(mask) else float("nan")
        out["int_mean_bg"] = float(np.mean(img[~mask])) if np.any(~mask) else float("nan")
        return out

    labels = seg.astype(np.int32)
    labs, counts = np.unique(labels, return_counts=True)
    total = int(counts.sum())
    k = int(labs.max() + 1)
    out["k"] = k

    class_stats = []
    for l, c in zip(labs, counts):
        m = labels == l
        class_stats.append(
            {
                "label": int(l),
                "count": int(c),
                "frac": float(c / total),
                "edge_mean": float(np.mean(emap[m])) if np.any(m) else float("nan"),
                "edge_std": float(np.std(emap[m])) if np.any(m) else float("nan"),
                "int_mean": float(np.mean(img[m])) if np.any(m) else float("nan"),
                "int_std": float(np.std(img[m])) if np.any(m) else float("nan"),
            }
        )
    boundary_label, boundary_info = infer_boundary_label(labels, emap, class_stats)

    excluded = set()
    if exclude_labels is not None:
        excluded.update(int(x) for x in exclude_labels)
    if boundary_label is not None:
        excluded.add(int(boundary_label))
    excluded_labels = sorted(excluded)

    if excluded_labels:
        slag_roi_mask = ~np.isin(labels, np.array(excluded_labels, dtype=np.int32))
    else:
        slag_roi_mask = np.ones(labels.shape, dtype=bool)

    boundary_mask = labels == int(boundary_label) if boundary_label is not None else np.zeros(labels.shape, dtype=bool)
    interior_mask = slag_roi_mask
    interior_total = int(np.sum(interior_mask))

    phase_stats = []
    for row in class_stats:
        if int(row["label"]) in excluded:
            continue
        phase_row = dict(row)
        phase_count_total = int(row["count"])
        phase_count = int(np.sum((labels == int(row["label"])) & interior_mask))
        phase_row["count_total"] = phase_count_total
        phase_row["count_slag_roi"] = phase_count
        # Backward-compatible aliases
        phase_row["count"] = phase_count_total
        phase_row["frac_total"] = float(phase_count_total / total) if total > 0 else float("nan")
        phase_row["frac"] = phase_row["frac_total"]
        if interior_total > 0:
            phase_row["frac_slag_roi"] = float(phase_count / interior_total)
        else:
            phase_row["frac_slag_roi"] = float("nan")
        # Backward-compatible key
        phase_row["frac_no_boundary"] = phase_row["frac_slag_roi"]
        phase_stats.append(phase_row)

    skel = skeletonize(boundary_mask)
    boundary_length_px = int(np.sum(skel))

    out["class_stats"] = class_stats
    out["boundary_label"] = int(boundary_label) if boundary_label is not None else None
    out["boundary_detected"] = bool(boundary_label is not None)
    out["boundary_inference"] = boundary_info
    out["roi_excluded_labels"] = excluded_labels
    out["slag_roi_pixels"] = interior_total
    out["slag_roi_area_frac"] = float(interior_total / float(total)) if total > 0 else float("nan")
    excluded_area_px = int(total - interior_total)
    out["excluded_area_pixels"] = excluded_area_px
    out["excluded_area_frac"] = float(excluded_area_px / float(total)) if total > 0 else float("nan")
    out["boundary_area_frac"] = float(np.mean(boundary_mask))
    out["interior_area_frac"] = float(interior_total / float(total)) if total > 0 else float("nan")
    out["interior_label_count"] = int(len(phase_stats))
    out["phase_stats"] = phase_stats
    # Backward-compatible alias
    out["interior_class_stats"] = phase_stats
    out["phase_frac_primary"] = "slag_roi"
    phase_frac_total_sum = float(np.sum([float(r["frac_total"]) for r in phase_stats])) if phase_stats else 0.0
    phase_frac_slag_roi_sum = (
        float(np.sum([float(r["frac_slag_roi"]) for r in phase_stats])) if interior_total > 0 and phase_stats else float("nan")
    )
    out["phase_frac_sum_total"] = phase_frac_total_sum
    out["phase_frac_sum_slag_roi"] = phase_frac_slag_roi_sum
    out["closure_boundary_plus_phase_total"] = float(out["boundary_area_frac"] + phase_frac_total_sum)
    out["closure_excluded_plus_phase_total"] = float(out["excluded_area_frac"] + phase_frac_total_sum)

    out["boundary_length_px"] = boundary_length_px
    out["boundary_length_density_per_px"] = float(boundary_length_px / float(total)) if total > 0 else float("nan")

    if k >= 3 and boundary_label is not None:
        shell_candidates = [r for r in class_stats if int(r["label"]) not in excluded]
        if shell_candidates:
            shell_label = int(max(shell_candidates, key=lambda r: float(r["edge_mean"]))["label"])
            shell_mask = (labels == shell_label) & interior_mask

            dt = distance_transform_edt(~boundary_mask)
            shell_dist = dt[shell_mask] if np.any(shell_mask) else np.array([], dtype=np.float32)

            out["shell_label"] = shell_label
            out["shell_area_frac"] = float(np.mean(shell_mask))
            if shell_dist.size > 0:
                out["shell_thickness_px_mean"] = float(np.mean(shell_dist))
                out["shell_thickness_px_median"] = float(np.median(shell_dist))
                out["shell_thickness_px_p90"] = float(np.quantile(shell_dist, 0.90))
            else:
                out["shell_thickness_px_mean"] = float("nan")
                out["shell_thickness_px_median"] = float("nan")
                out["shell_thickness_px_p90"] = float("nan")

    return out


# -----------------------------
# Save / visualization
# -----------------------------
def make_discrete_label_cmap(n_labels: int):
    """High-contrast discrete colormap for integer labels [0, n_labels-1]."""
    if n_labels < 1:
        raise ValueError(f"n_labels must be >= 1. Got {n_labels}")

    if n_labels <= len(SEM_LABEL_COLORS):
        colors = SEM_LABEL_COLORS[:n_labels]
    else:
        extra_n = n_labels - len(SEM_LABEL_COLORS)
        extra = plt.cm.tab20(np.linspace(0.0, 1.0, max(extra_n, 1)))[:extra_n]
        colors = np.vstack([SEM_LABEL_COLORS, extra])

    cmap = mcolors.ListedColormap(colors, name="sem_labels")
    boundaries = np.arange(-0.5, n_labels + 0.5, 1.0)
    norm = mcolors.BoundaryNorm(boundaries, ncolors=n_labels)
    return cmap, norm


def save_quicklook(img, emap, seg, outdir, mode):
    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots(
        1,
        4,
        figsize=(14, 5),
        gridspec_kw={"width_ratios": [1, 1, 1, 0.05], "wspace": 0.08},
        constrained_layout=True,
    )

    ax0, ax1, ax2, cax = ax

    ax0.imshow(img, cmap="gray")
    ax0.set_title("Cropped SEM")
    ax0.axis("off")

    ax1.imshow(emap, cmap="gray")
    ax1.set_title("Edge map")
    ax1.axis("off")

    if mode == "otsu":
        ax2.imshow(seg.astype(bool), cmap="gray")
        ax2.set_title("Segmentation (Otsu)")
        ax2.axis("off")
        cax.axis("off")
    else:
        labels = seg.astype(np.int32)
        n_labels = int(np.max(labels) + 1)
        cmap, norm = make_discrete_label_cmap(n_labels)

        im = ax2.imshow(labels, cmap=cmap, norm=norm, interpolation="nearest")
        ax2.set_title("Segmentation (k-means)")
        ax2.axis("off")
        fig.colorbar(im, cax=cax, ticks=np.arange(n_labels))

    fig.savefig(os.path.join(outdir, "quicklook.png"), dpi=300)
    plt.close(fig)


def save_phase_only_png(seg, metrics: dict, outdir: str, mode: str):
    """Save k-means phase map with excluded ROI labels hidden as black."""
    if mode != "kmeans":
        return

    labels = seg.astype(np.int32)
    excluded_labels = sorted(int(x) for x in metrics.get("roi_excluded_labels", []))
    excluded_set = set(excluded_labels)

    phase_rows = metrics.get("phase_stats", [])
    phase_labels = [int(row["label"]) for row in phase_rows if int(row["label"]) not in excluded_set]
    if not phase_labels:
        phase_labels = [int(l) for l in np.unique(labels) if int(l) not in excluded_set]

    if not phase_labels:
        return

    phase_labels = sorted(phase_labels)
    label_to_idx = {lab: i for i, lab in enumerate(phase_labels)}

    phase_vis = np.full(labels.shape, np.nan, dtype=np.float32)
    for lab, idx in label_to_idx.items():
        phase_vis[labels == lab] = float(idx)

    cmap, norm = make_discrete_label_cmap(len(phase_labels))
    cmap = mcolors.ListedColormap(cmap.colors.copy())
    cmap.set_bad(color=(0.0, 0.0, 0.0, 1.0))  # excluded ROI labels shown as black

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
    im = ax.imshow(
        np.ma.masked_invalid(phase_vis),
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    ax.set_title("Phase map (slag ROI)")
    ax.axis("off")

    ticks = np.arange(len(phase_labels))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=ticks)
    cbar.ax.set_yticklabels([f"label {lab}" for lab in phase_labels])
    cbar.set_label("phase")

    fig.savefig(os.path.join(outdir, "phase_only.png"), dpi=300)
    plt.close(fig)


def build_phase_summary_rows(metrics: dict) -> list[dict]:
    """Build one row per label so phase metrics are easy to inspect."""
    class_stats = metrics.get("class_stats", [])
    if not class_stats:
        return []

    phase_map = {int(r["label"]): r for r in metrics.get("phase_stats", [])}
    boundary_label = metrics.get("boundary_label")
    excluded = set(int(x) for x in metrics.get("roi_excluded_labels", []))

    rows = []
    for row in sorted(class_stats, key=lambda r: int(r["label"])):
        label = int(row["label"])
        phase_row = phase_map.get(label)
        if phase_row is None:
            count_slag_roi = 0
            frac_slag_roi = 0.0
            frac_no_boundary = 0.0
        else:
            count_slag_roi = int(phase_row.get("count_slag_roi", phase_row.get("count", 0)))
            frac_slag_roi = float(phase_row.get("frac_slag_roi", float("nan")))
            frac_no_boundary = float(phase_row.get("frac_no_boundary", frac_slag_roi))

        rows.append(
            {
                "phase_label": label,
                "count_total": int(row["count"]),
                "frac_total": float(row["frac"]),
                "count_slag_roi": count_slag_roi,
                "frac_slag_roi": frac_slag_roi,
                "frac_no_boundary": frac_no_boundary,
                "is_boundary": int(boundary_label is not None and int(boundary_label) == label),
                "is_excluded_from_slag_roi": int(label in excluded),
                "edge_mean": float(row["edge_mean"]),
                "edge_std": float(row["edge_std"]),
                "int_mean": float(row["int_mean"]),
                "int_std": float(row["int_std"]),
            }
        )
    return rows


def save_phase_summary_csv(metrics: dict, outdir: str) -> None:
    rows = build_phase_summary_rows(metrics)
    if not rows:
        return
    csv_path = os.path.join(outdir, "phase_summary.csv")
    fieldnames = [
        "phase_label",
        "count_total",
        "frac_total",
        "count_slag_roi",
        "frac_slag_roi",
        "frac_no_boundary",
        "is_boundary",
        "is_excluded_from_slag_roi",
        "edge_mean",
        "edge_std",
        "int_mean",
        "int_std",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def save_outputs(img, emap, seg, metrics: dict, outdir: str, mode: str):
    os.makedirs(outdir, exist_ok=True)
    phase_summary_rows = build_phase_summary_rows(metrics)

    np.save(os.path.join(outdir, "img_cropped.npy"), img)
    np.save(os.path.join(outdir, "edge_map.npy"), emap)
    np.save(os.path.join(outdir, f"seg_{mode}.npy"), seg)

    with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(outdir, "stats.txt"), "w", encoding="utf-8") as f:
        f.write("SEM segmentation stats\n\n")
        for k, v in metrics.items():
            if k in {"class_stats", "phase_stats", "interior_class_stats", "boundary_inference"}:
                continue
            f.write(f"{k}: {v}\n")

        if "class_stats" in metrics:
            f.write("\nclass_stats:\n")
            for row in metrics["class_stats"]:
                f.write(
                    f"  label {row['label']}: frac={row['frac']:.6f}, "
                    f"edge={row['edge_mean']:.6f}±{row['edge_std']:.6f}, "
                    f"int={row['int_mean']:.6f}±{row['int_std']:.6f}\n"
                )

        if "phase_stats" in metrics:
            f.write("\nphase_stats (slag ROI denominator):\n")
            for row in metrics["phase_stats"]:
                f.write(
                    f"  label {row['label']}: count_total={row['count_total']}, "
                    f"count_slag_roi={row['count_slag_roi']}, "
                    f"frac_total={row['frac_total']:.6f}, "
                    f"frac_slag_roi={row['frac_slag_roi']:.6f}, "
                    f"edge={row['edge_mean']:.6f}±{row['edge_std']:.6f}, "
                    f"int={row['int_mean']:.6f}±{row['int_std']:.6f}\n"
                )
        if phase_summary_rows:
            f.write("\nphase_summary (grouped by phase label):\n")
            for row in phase_summary_rows:
                f.write(
                    f"  phase {row['phase_label']}: count_total={row['count_total']}, "
                    f"count_slag_roi={row['count_slag_roi']}, "
                    f"frac_total={row['frac_total']:.6f}, "
                    f"frac_no_boundary={row['frac_no_boundary']:.6f}, "
                    f"excluded={row['is_excluded_from_slag_roi']}, "
                    f"boundary={row['is_boundary']}\n"
                )

    save_quicklook(img, emap, seg, outdir, mode)
    save_phase_only_png(seg, metrics, outdir, mode)
    save_phase_summary_csv(metrics, outdir)


def flatten_metrics_for_csv(metrics: dict) -> dict:
    flat = {}
    phase_summary_rows = build_phase_summary_rows(metrics)
    for k, v in metrics.items():
        if k == "class_stats":
            flat["class_stats"] = json.dumps(v, ensure_ascii=False)
        elif k == "phase_stats":
            flat["phase_stats"] = json.dumps(v, ensure_ascii=False)
        elif k in {
            "interior_class_stats",
            "roi_excluded_labels",
            "user_exclude_labels",
            "boundary_inference",
        }:
            flat[k] = json.dumps(v, ensure_ascii=False)
        else:
            flat[k] = v

    if phase_summary_rows:
        flat["phase_summary"] = json.dumps(phase_summary_rows, ensure_ascii=False)
        for row in phase_summary_rows:
            label = int(row["phase_label"])
            flat[f"phase{label}_count_total"] = int(row["count_total"])
            flat[f"phase{label}_count_slag_roi"] = int(row["count_slag_roi"])
            flat[f"phase{label}_frac_total"] = float(row["frac_total"])
            flat[f"phase{label}_frac_slag_roi"] = float(row["frac_slag_roi"])
            flat[f"phase{label}_frac_no_boundary"] = float(row["frac_no_boundary"])
            # Primary phase fraction for quick comparison (sum ~= 1 over slag ROI phases)
            flat[f"phase{label}_frac"] = float(row["frac_slag_roi"])
            flat[f"phase{label}_excluded"] = int(row["is_excluded_from_slag_roi"])
            flat[f"phase{label}_is_boundary"] = int(row["is_boundary"])
            # Backward-compatible alias
            flat[f"phase{label}_count"] = int(row["count_total"])
    return flat


def sort_summary_keys(keys: set[str]) -> list[str]:
    """
    Keep general fields first, then group per-phase fields in a readable order.
    """
    phase_field_order = {
        "count_total": 0,
        "count_slag_roi": 1,
        "frac_total": 2,
        "frac_slag_roi": 3,
        "frac_no_boundary": 4,
        "frac": 5,
        "excluded": 6,
        "is_boundary": 7,
        "count": 8,
    }

    def _key_fn(key: str):
        m = re.fullmatch(r"phase(\d+)_(.+)", key)
        if m is None:
            return (0, key)
        label = int(m.group(1))
        field = m.group(2)
        return (1, label, phase_field_order.get(field, 100), field)

    return sorted(keys, key=_key_fn)


# -----------------------------
# Python-first config & API
# -----------------------------
@dataclass
class RunConfig:
    # Shared
    mode: str = "kmeans"            # "otsu" or "kmeans"
    method: str = "sobel"           # "sobel" or "laplacian"
    win: int = 41
    crop_bottom_frac: float = 0.0625
    min_dim: int = 0
    min_size: int = 0              # otsu only
    seed: int = 0

    # kmeans only
    k: int = 3
    kmeans_features: str = "edge+intensity"  # "edge" or "edge+intensity"
    kmeans_w_edge: float = 1.0
    kmeans_w_int: float = 1.0
    kmeans_min_size: int = 4
    roi_exclude_labels: tuple[int, ...] = ()


def _validate_config(cfg: RunConfig) -> None:
    if cfg.mode not in ("otsu", "kmeans"):
        raise ValueError(f"mode must be 'otsu' or 'kmeans'. Got {cfg.mode}")
    if cfg.method not in ("sobel", "laplacian"):
        raise ValueError(f"method must be 'sobel' or 'laplacian'. Got {cfg.method}")
    if cfg.win is None or int(cfg.win) < 1:
        raise ValueError(f"win must be >= 1. Got {cfg.win}")
    if not (0.0 <= float(cfg.crop_bottom_frac) < 1.0):
        raise ValueError(f"crop_bottom_frac must be in [0,1). Got {cfg.crop_bottom_frac}")
    if cfg.min_dim is None or int(cfg.min_dim) < 0:
        raise ValueError(f"min_dim must be >= 0. Got {cfg.min_dim}")
    if cfg.seed is None:
        raise ValueError("seed must not be None")
    if cfg.mode == "otsu":
        if cfg.min_size is None or int(cfg.min_size) < 0:
            raise ValueError(f"min_size must be >= 0. Got {cfg.min_size}")
    if cfg.mode == "kmeans":
        if cfg.k is None or int(cfg.k) < 2:
            raise ValueError(f"k must be >= 2 for kmeans. Got {cfg.k}")
        if cfg.kmeans_features not in ("edge", "edge+intensity"):
            raise ValueError(
                "kmeans_features must be 'edge' or 'edge+intensity'. "
                f"Got {cfg.kmeans_features}"
            )
        if cfg.kmeans_w_edge is None or float(cfg.kmeans_w_edge) <= 0:
            raise ValueError(f"kmeans_w_edge must be > 0. Got {cfg.kmeans_w_edge}")
        if cfg.kmeans_w_int is None or float(cfg.kmeans_w_int) <= 0:
            raise ValueError(f"kmeans_w_int must be > 0. Got {cfg.kmeans_w_int}")
        if cfg.kmeans_min_size is None or int(cfg.kmeans_min_size) < 0:
            raise ValueError(f"kmeans_min_size must be >= 0. Got {cfg.kmeans_min_size}")
        if cfg.roi_exclude_labels is None:
            raise ValueError("roi_exclude_labels must not be None")
        if any(int(x) < 0 for x in cfg.roi_exclude_labels):
            raise ValueError(
                f"roi_exclude_labels must be >= 0. Got {cfg.roi_exclude_labels}"
            )


def run(infile: str, cfg: RunConfig, outdir: Optional[str] = None) -> dict:
    """
    Python-first API.

    Usage (in Python):
        import sem_segmentation_mac as sem
        cfg = sem.RunConfig(mode="kmeans", k=4, win=25)
        metrics = sem.run("image.tif", cfg)
    """
    _validate_config(cfg)

    infile_path = Path(infile).expanduser().resolve()
    if not infile_path.exists():
        raise FileNotFoundError(f"infile not found: {infile_path}")

    # Default outdir: <infile_stem>_out next to input file
    if outdir is None or str(outdir).strip() == "":
        outdir_path = infile_path.parent / f"{infile_path.stem}_out"
    else:
        outdir_path = Path(outdir).expanduser()

    # --- pipeline (reuse your existing functions) ---
    img = load_image(str(infile_path))
    img = crop_bottom_fraction(img, frac=float(cfg.crop_bottom_frac))
    img = maybe_resize_min(img, int(cfg.min_dim))

    emap = edge_map(img, method=str(cfg.method), win=int(cfg.win))

    kinfo = {}
    if cfg.mode == "otsu":
        seg = seg_otsu(emap)
        seg = postprocess_binary(seg, min_size=int(cfg.min_size))
    else:
        seg, kinfo = seg_kmeans(
            img,
            emap,
            k=int(cfg.k),
            seed=int(cfg.seed),
            features=str(cfg.kmeans_features),
            w_edge=float(cfg.kmeans_w_edge),
            w_int=float(cfg.kmeans_w_int),
        )
        seg, n_reassigned = postprocess_kmeans_labels(seg, min_size=int(cfg.kmeans_min_size))
        kinfo["kmeans_min_size"] = int(cfg.kmeans_min_size)
        kinfo["kmeans_reassigned_px"] = int(n_reassigned)

    metrics = compute_metrics(
        img,
        emap,
        seg,
        mode=str(cfg.mode),
        exclude_labels=list(cfg.roi_exclude_labels),
    )

    # Metadata
    metrics["infile"] = infile_path.name
    metrics["method"] = str(cfg.method)
    metrics["win"] = int(cfg.win)
    metrics["crop_bottom_frac"] = float(cfg.crop_bottom_frac)
    metrics["min_dim"] = int(cfg.min_dim)
    metrics["min_size"] = int(cfg.min_size)
    metrics["seed"] = int(cfg.seed)
    metrics["user_exclude_labels"] = [int(x) for x in cfg.roi_exclude_labels]

    if cfg.mode == "kmeans":
        metrics["k"] = int(cfg.k)
        metrics["kmeans_features"] = str(cfg.kmeans_features)
        metrics["kmeans_w_edge"] = float(cfg.kmeans_w_edge)
        metrics["kmeans_w_int"] = float(cfg.kmeans_w_int)
        metrics["kmeans_min_size"] = int(cfg.kmeans_min_size)
        metrics.update(kinfo)

    save_outputs(img, emap, seg, metrics, str(outdir_path), mode=str(cfg.mode))
    return metrics


# -----------------------------
# CLI / auto mode (thin wrappers around Python API)
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="SEM edge-based segmentation toolkit (run, sweep, and auto mode)."
    )

    p.add_argument("cmd", nargs="?", choices=["run", "sweep"], help="Optional command")

    # Shared options
    p.add_argument("--mode", choices=["otsu", "kmeans"], default="kmeans")
    p.add_argument("--method", choices=["laplacian", "sobel"], default="sobel")
    p.add_argument("--win", type=int, default=41)
    p.add_argument("--crop_bottom_frac", type=float, default=0.0625)
    p.add_argument("--min_dim", type=int, default=0)
    p.add_argument("--min_size", type=int, default=0, help="(otsu only) remove small objects")
    p.add_argument("--seed", type=int, default=0)

    # run mode
    p.add_argument("--infile", default="")
    p.add_argument("--outdir", default="")
    p.add_argument("--k", type=int, default=3, help="(kmeans only) number of clusters")
    p.add_argument(
        "--kmeans_features",
        choices=["edge", "edge+intensity"],
        default="edge+intensity",
        help="(kmeans only) feature space for clustering",
    )
    p.add_argument(
        "--kmeans_w_edge",
        type=float,
        default=1.0,
        help="(kmeans only) post-standardization weight for edge feature",
    )
    p.add_argument(
        "--kmeans_w_int",
        type=float,
        default=1.0,
        help="(kmeans only) post-standardization weight for intensity feature",
    )
    p.add_argument(
        "--kmeans_min_size",
        type=int,
        default=4,
        help="(kmeans only) remove islands smaller than this size, then nearest-label reassign",
    )
    p.add_argument(
        "--exclude_labels",
        type=int,
        nargs="*",
        default=[],
        help="(kmeans only) labels to exclude from slag ROI denominator",
    )

    # sweep mode (keep as-is; uses existing run_sweep_mode)
    p.add_argument("--input", default="", help='Glob pattern or directory (e.g. "./data/*.tif")')
    p.add_argument("--outroot", default="", help="Root output directory")
    p.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--wins", type=int, nargs="+", default=[11, 21, 41, 61, 81])

    return p.parse_args()


def args_to_config(args) -> RunConfig:
    return RunConfig(
        mode=str(args.mode),
        method=str(args.method),
        win=int(args.win),
        crop_bottom_frac=float(args.crop_bottom_frac),
        min_dim=int(args.min_dim),
        min_size=int(args.min_size),
        seed=int(args.seed),
        k=int(args.k),
        kmeans_features=str(args.kmeans_features),
        kmeans_w_edge=float(args.kmeans_w_edge),
        kmeans_w_int=float(args.kmeans_w_int),
        kmeans_min_size=int(args.kmeans_min_size),
        roi_exclude_labels=tuple(sorted(set(int(x) for x in args.exclude_labels))),
    )


def auto_find_same_stem_tif(script_path: Path) -> Path:
    """
    Auto mode priority:
    1) TIFF with same stem as this script in same folder.
    2) If exactly one TIFF exists in folder, use it.
    """
    folder = script_path.parent
    stem = script_path.stem

    same_stem = []
    for ext in (".tif", ".tiff", ".TIF", ".TIFF"):
        p = folder / f"{stem}{ext}"
        if p.exists():
            same_stem.append(p)

    if same_stem:
        return same_stem[0]

    candidates = sorted(
        [
            *folder.glob("*.tif"),
            *folder.glob("*.tiff"),
            *folder.glob("*.TIF"),
            *folder.glob("*.TIFF"),
        ]
    )

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No TIFF found in {folder}. Put a .tif/.tiff file next to this script."
        )

    names = "\n".join(f"- {p.name}" for p in candidates)
    raise ValueError(
        "Multiple TIFF files found. Either keep one TIFF, rename one to match script name, "
        f"or pass --infile explicitly:\n{names}"
    )


def run_auto_mode(args):
    cfg = args_to_config(args)
    script_path = Path(__file__).resolve()
    infile = auto_find_same_stem_tif(script_path)

    # Output next to script, like original behavior
    outdir = script_path.parent / f"{infile.stem}_out"
    metrics = run(str(infile), cfg, outdir=str(outdir))

    print("[DONE] auto")
    print(f"Input: {infile}")
    print(f"Output: {outdir}")
    print(json.dumps(flatten_metrics_for_csv(metrics), ensure_ascii=False, indent=2))


def run_run_mode(args):
    if not args.infile:
        raise ValueError("run mode requires --infile")

    cfg = args_to_config(args)

    if args.outdir:
        outdir = args.outdir
    else:
        p = Path(args.infile)
        outdir = str(p.parent / f"{p.stem}_out")

    metrics = run(args.infile, cfg, outdir=outdir)
    print("[DONE] run")
    print(json.dumps(flatten_metrics_for_csv(metrics), ensure_ascii=False, indent=2))

def expand_inputs(inp: str):
    """Accept a directory or a glob pattern and return sorted tif/tiff files."""
    if os.path.isdir(inp):
        files = glob.glob(os.path.join(inp, "*.tif")) + glob.glob(os.path.join(inp, "*.tiff"))
        files += glob.glob(os.path.join(inp, "*.TIF")) + glob.glob(os.path.join(inp, "*.TIFF"))
        return sorted(list(set(files)))
    files = glob.glob(inp)
    return sorted(files)


def run_sweep_mode(args):
    """
    Sweep over files and (k, win) combinations.
    Output layout:
      <outroot>/<base>/mode_kmeans/k_<k>/win_<win>/...
      <outroot>/<base>/mode_otsu/win_<win>/...
    Writes <outroot>/summary.csv
    """
    if not args.input:
        raise ValueError("sweep mode requires --input")

    outroot = args.outroot if args.outroot else "sweep_out"
    os.makedirs(outroot, exist_ok=True)

    files = expand_inputs(args.input)
    if not files:
        raise FileNotFoundError(f"No input files found for: {args.input}")

    summary_rows = []

    # base config from args
    cfg0 = args_to_config(args)

    for infile in files:
        base = os.path.splitext(os.path.basename(infile))[0]

        for win in args.wins:
            win = int(win)

            if cfg0.mode == "otsu":
                cfg = RunConfig(
                    mode="otsu",
                    method=cfg0.method,
                    win=win,
                    crop_bottom_frac=cfg0.crop_bottom_frac,
                    min_dim=cfg0.min_dim,
                    min_size=cfg0.min_size,
                    seed=cfg0.seed,
                    k=cfg0.k,  # unused in otsu
                    kmeans_features=cfg0.kmeans_features,
                    kmeans_w_edge=cfg0.kmeans_w_edge,
                    kmeans_w_int=cfg0.kmeans_w_int,
                    kmeans_min_size=cfg0.kmeans_min_size,
                    roi_exclude_labels=cfg0.roi_exclude_labels,
                )
                outdir = os.path.join(outroot, base, "mode_otsu", f"win_{win}")
                metrics = run(infile, cfg, outdir=outdir)
                row = flatten_metrics_for_csv(metrics)
                row.update({"base": base, "outdir": outdir})
                summary_rows.append(row)

            else:
                for k in args.ks:
                    k = int(k)
                    cfg = RunConfig(
                        mode="kmeans",
                        method=cfg0.method,
                        win=win,
                        crop_bottom_frac=cfg0.crop_bottom_frac,
                        min_dim=cfg0.min_dim,
                        min_size=cfg0.min_size,  # unused in kmeans
                        seed=cfg0.seed,
                        k=k,
                        kmeans_features=cfg0.kmeans_features,
                        kmeans_w_edge=cfg0.kmeans_w_edge,
                        kmeans_w_int=cfg0.kmeans_w_int,
                        kmeans_min_size=cfg0.kmeans_min_size,
                        roi_exclude_labels=cfg0.roi_exclude_labels,
                    )
                    outdir = os.path.join(outroot, base, "mode_kmeans", f"k_{k}", f"win_{win}")
                    metrics = run(infile, cfg, outdir=outdir)
                    row = flatten_metrics_for_csv(metrics)
                    row.update({"base": base, "outdir": outdir})
                    summary_rows.append(row)

    csv_path = os.path.join(outroot, "summary.csv")
    keys = sort_summary_keys({k for row in summary_rows for k in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    print("[DONE] sweep")
    print(f"Summary saved: {csv_path}")

def main():
    args = parse_args()

    # No command -> auto mode
    if args.cmd is None:
        run_auto_mode(args)
        return

    if args.cmd == "run":
        run_run_mode(args)
        return

    if args.cmd == "sweep":
        run_sweep_mode(args)
        return


if __name__ == "__main__":
    main()
