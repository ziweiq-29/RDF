#!/usr/bin/env python3
"""Plot RDF curves (g1/g2) with x=pair distance and y=g(r).

Modes:
1) --from_npy: read existing rdf_raw.npy / rdf_dec.npy directly
2) default: compute RDF from raw/decompressed coordinates, then plot
3) --quad: four PNGs (raw vs decompressed per compressor) plus one rdf_original_only_<dataset>.png
   (raw curve only), using QUAD_RDF_SPECS / --quad_specs / --quad_compressors + --quad_error_bounds.
   Overlay filenames include mean PSNR and mean compression_ratio from outputs/STANDARD/EXAALT
   (see --exaalt_suite).
"""

import argparse
import glob
import os
import csv
import re
from typing import Any, Dict, List, Optional, Tuple
import subprocess
import numpy as np
import matplotlib.pyplot as plt


# Quad overlay: four compressors, each with its own relative error bound (edit here or use --quad_specs).
QUAD_RDF_SPECS: List[Tuple[str, str]] = [
    ("zfp", "1e-2"),
    ("mgard", "1e-2"),
    ("sz3", "1e-2"),
    ("sperr", "1e-2"),
]

DEFAULT_PAIR_DISTANCE_XMAX = 8.0

# Publication-style sizes for quad RDF figures (axis labels / ticks).
RDF_AXIS_LABEL_FONTSIZE = 20
RDF_TICK_LABEL_FONTSIZE = 18


def _apply_rdf_axis_style(ax=None):
    """Axis label and major tick label sizes; no title (callers omit plt.title)."""
    if ax is None:
        ax = plt.gca()
    ax.tick_params(axis="both", which="major", labelsize=RDF_TICK_LABEL_FONTSIZE)

ERROR_BOUNDS = [
    "1e-6", "5e-6", "1e-5", "5e-5", "1e-4", "5e-4", "1e-3", "5e-3", "1e-2", "5e-2", "1e-1"
]


def _norm_eb(s: str) -> str:
    t = (s or "").strip()
    if not t:
        return ""
    try:
        return f"{float(t):.12g}"
    except (ValueError, TypeError):
        return t


_EB_NORM_TO_CANON = {_norm_eb(eb): eb for eb in ERROR_BOUNDS}
# Same filter as get_scatter_plots/RDF/plot_rdf.py gather_standard_xyz_aggregates().
_ALLOWED_EB_NORM = set(_EB_NORM_TO_CANON.keys())


def to_float(x, default=np.nan):
    if x is None or (isinstance(x, str) and x.strip() in ("", "<empty>")):
        return default
    try:
        return float(x)
    except (ValueError, TypeError):
        return default


def _safe_filename_piece(s: str) -> str:
    s = str(s)
    s = s.replace(os.sep, "_")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "NA"


def _standard_dataset_stem(dataset_name: str) -> str:
    """Folder names use dataset1-5423x3137, not dataset1-5423x3137_f32."""
    s = (dataset_name or "").strip()
    if s.endswith("_f32"):
        s = s[: -len("_f32")]
    return s


def _read_csv_dicts(path: str):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pick_one(candidates):
    candidates = [c for c in candidates if c]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    candidates = sorted(candidates)
    raise FileNotFoundError(
        "Multiple candidate npy files found; please specify explicit path. Candidates:\n  "
        + "\n  ".join(candidates)
    )


def auto_find_rdf_npy_paths(search_dir: str):
    """
    Auto-find rdf_raw and rdf_dec npy files in a directory.
    Priority:
      1) exact rdf_raw.npy / rdf_dec.npy
      2) any file matching rdf_raw*.npy / rdf_dec*.npy
    """
    if not search_dir:
        search_dir = "."
    if not os.path.isdir(search_dir):
        raise FileNotFoundError(f"RDF npy search directory does not exist: {search_dir}")

    exact_raw = os.path.join(search_dir, "rdf_raw.npy")
    exact_dec = os.path.join(search_dir, "rdf_dec.npy")
    if os.path.isfile(exact_raw) and os.path.isfile(exact_dec):
        return exact_raw, exact_dec

    raw_candidates = []
    dec_candidates = []
    for name in os.listdir(search_dir):
        if not name.endswith(".npy"):
            continue
        if name.startswith("rdf_raw"):
            raw_candidates.append(os.path.join(search_dir, name))
        elif name.startswith("rdf_dec"):
            dec_candidates.append(os.path.join(search_dir, name))

    raw = exact_raw if os.path.isfile(exact_raw) else _pick_one(raw_candidates)
    dec = exact_dec if os.path.isfile(exact_dec) else _pick_one(dec_candidates)

    if raw is None:
        raise FileNotFoundError(
            f"Could not find raw RDF npy in {search_dir} (expected rdf_raw.npy or rdf_raw*.npy)"
        )
    if dec is None:
        raise FileNotFoundError(
            f"Could not find dec RDF npy in {search_dir} (expected rdf_dec.npy or rdf_dec*.npy)"
        )
    return raw, dec


def aggregate_standard_xyz_metrics_like_plot_rdf(
    compressor_name: str,
    error_bound: str,
    dataset_name: str,
    exaalt_suite: str = "copper",
) -> Optional[Dict[str, Any]]:
    """
    One STANDARD aggregate per (compressor, dataset, error_bound), matching
    compression_framework/get_scatter_plots/RDF/plot_rdf.py gather_standard_xyz_aggregates():

    - Numeric columns (including psnr): arithmetic mean over x, y, z rows.
    - compression_ratio: sum(uncompressed_size) / sum(compressed_size) across axes (preferred).
    - bit_rate: 32.0 / compression_ratio when sizes are valid; else mean(axis bit_rate).
    - error_bound must normalize to an entry in ERROR_BOUNDS (same allowed set as plot_rdf.py).
    """
    if compressor_name in ("", "NA") or dataset_name in ("", "NA") or error_bound in ("", "NA"):
        return None
    if not compressor_name or not dataset_name or not error_bound:
        return None

    eb_n = _norm_eb(error_bound)
    if not eb_n or eb_n not in _ALLOWED_EB_NORM:
        return None

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_cf = os.path.abspath(os.path.join(script_dir, "..", "compression_framework"))
    std_root = os.path.join(base_cf, "outputs", "STANDARD", "EXAALT")
    dataset_key = _standard_dataset_stem(dataset_name)
    suite = (exaalt_suite or "copper").strip().lower()
    if suite not in ("copper", "helium"):
        suite = "copper"

    rows: List[dict] = []
    for axis in ("x", "y", "z"):
        dpath = os.path.join(
            std_root, f"SDRBENCH-exaalt-{suite}_{dataset_key}_{axis}_f32_dat"
        )
        csv_path = os.path.join(dpath, f"{compressor_name}_standard.csv")
        if not os.path.isfile(csv_path):
            return None
        csv_rows = _read_csv_dicts(csv_path)
        row_ax = None
        for row in csv_rows:
            row_eb = _norm_eb(row.get("error_bound", ""))
            if row_eb not in _ALLOWED_EB_NORM:
                continue
            if row_eb != eb_n:
                continue
            row_ax = dict(row)
            break
        if row_ax is None:
            return None
        row_ax["_axis"] = axis
        rows.append(row_ax)

    axes = {r.get("_axis") for r in rows}
    if axes != {"x", "y", "z"}:
        return None

    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())

    out: Dict[str, Any] = {}
    for k in sorted(all_keys):
        if k in ("_axis", "compressor name", "input", "error_bound"):
            continue
        vals = [to_float(r.get(k)) for r in rows]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            out[k] = float(np.mean(vals))

    comp_sizes = [to_float(r.get("compressed_size")) for r in rows]
    uncomp_sizes = [to_float(r.get("uncompressed_size")) for r in rows]
    if all(np.isfinite(v) and v > 0 for v in comp_sizes + uncomp_sizes):
        total_comp = float(np.sum(comp_sizes))
        total_uncomp = float(np.sum(uncomp_sizes))
        cr_total = total_uncomp / total_comp
        out["compression_ratio"] = cr_total
        out["bit_rate"] = 32.0 / cr_total
    else:
        brs = [to_float(r.get("bit_rate")) for r in rows]
        brs = [v for v in brs if np.isfinite(v) and v > 0]
        out["bit_rate"] = float(np.mean(brs)) if brs else float("nan")

    out["compressor name"] = compressor_name
    out["dataset"] = dataset_key
    out["error_bound"] = _EB_NORM_TO_CANON.get(eb_n, eb_n)
    return out


def compute_cr_total_from_standard_csv(
    compressor_name: str,
    error_bound: str,
    dataset_name: str,
    exaalt_suite: str = "copper",
) -> Optional[float]:
    """compression_ratio from aggregate_standard_xyz_metrics_like_plot_rdf (sum-size CR)."""
    agg = aggregate_standard_xyz_metrics_like_plot_rdf(
        compressor_name, error_bound, dataset_name, exaalt_suite=exaalt_suite
    )
    if agg is None:
        return None
    cr = agg.get("compression_ratio")
    if cr is None or not np.isfinite(cr):
        return None
    return float(cr)


def compute_mean_psnr_from_standard_csv(
    compressor_name: str,
    error_bound: str,
    dataset_name: str,
    exaalt_suite: str = "copper",
) -> Optional[float]:
    """PSNR from aggregate_standard_xyz_metrics_like_plot_rdf (mean over x,y,z)."""
    agg = aggregate_standard_xyz_metrics_like_plot_rdf(
        compressor_name, error_bound, dataset_name, exaalt_suite=exaalt_suite
    )
    if agg is None:
        return None
    p = agg.get("psnr")
    if p is None or not np.isfinite(p):
        return None
    return float(p)


def lookup_wasserstein_from_rdf_csv(
    compressor_name: str,
    error_bound: str,
    dataset_name: str,
    explicit_csv: Optional[str] = None,
) -> Optional[float]:
    """
    Read wasserstein_distance from compression_framework/outputs/RDF/**/{dataset}_f32/{compressor}_rdf.csv.
    This matches the QOI reported by the RDF pressio pipeline (not a recomputed EMD on plotted g(r)).
    """
    if compressor_name in ("", "NA") or dataset_name in ("", "NA") or error_bound in ("", "NA"):
        return None
    if not compressor_name or not dataset_name or not error_bound:
        return None

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_cf = os.path.abspath(os.path.join(script_dir, "..", "compression_framework"))
    rdf_root = os.path.join(base_cf, "outputs", "RDF")
    if not os.path.isdir(rdf_root):
        return None

    ds_base = dataset_name.strip()
    if ds_base.endswith("_f32"):
        folder_ds = ds_base
        ds_key = ds_base[: -len("_f32")]
    else:
        ds_key = ds_base
        folder_ds = f"{ds_base}_f32"

    if explicit_csv:
        csv_path = os.path.abspath(explicit_csv)
        if not os.path.isfile(csv_path):
            return None
    else:
        pattern = os.path.join(rdf_root, "**", folder_ds, f"{compressor_name}_rdf.csv")
        matches = sorted(glob.glob(pattern, recursive=True))
        if not matches:
            return None
        if len(matches) > 1:
            # Prefer main RDF bundles under outputs/RDF/<suite>/… over app_eval_sec copies (metrics differ).
            primary = [
                m
                for m in matches
                if "app_eval_sec" not in m.replace("\\", "/")
            ]
            pool = primary if primary else matches
            csv_path = sorted(pool)[0]
        else:
            csv_path = matches[0]

    eb_n = _norm_eb(error_bound)
    rows = _read_csv_dicts(csv_path)
    for row in rows:
        inp = (row.get("input") or "").strip()
        if inp != ds_key:
            continue
        if _norm_eb(row.get("error_bound", "")) != eb_n:
            continue
        w = to_float(row.get("wasserstein_distance"))
        if np.isfinite(w):
            return float(w)
    return None


def _prefix_from_any_axis_path(prefix_or_axis_path: str) -> str:
    """
    Accept either:
      - dataset prefix: /path/to/dataset1-7852x1037
      - axis file path: /path/to/dataset1-7852x1037.x.f32.dat
    and return the normalized prefix.
    """
    p = (prefix_or_axis_path or "").strip()
    if not p:
        return p
    for suf in (".x.f32.dat", ".y.f32.dat", ".z.f32.dat"):
        if p.endswith(suf):
            return p[: -len(suf)]
    return p.rstrip(".")


def pressio_decompress_xyz(
    *,
    pressio_bin: str,
    raw_prefix: str,
    nt: int,
    na: int,
    compressor: str,
    rel: float,
    pressio_opts: list[str],
    out_dir: str,
    out_tag: str = "",
) -> tuple[str, str, str]:
    """
    Run pressio for x/y/z and write decompressed float32 arrays to files via -W.
    Returns (dec_x_path, dec_y_path, dec_z_path).
    """
    raw_prefix = _prefix_from_any_axis_path(raw_prefix)
    expected_bytes = int(nt) * int(na) * 4
    dec_paths = {}

    for axis in ("x", "y", "z"):
        inp = f"{raw_prefix}.{axis}.f32.dat"
        if not os.path.isfile(inp):
            raise FileNotFoundError(f"Missing raw axis file: {inp}")

    tag = _safe_filename_piece(out_tag) if out_tag else ""
    for axis in ("x", "y", "z"):
        inp = f"{raw_prefix}.{axis}.f32.dat"
        fname = f"dec_{tag}_{axis}.f32.bin" if tag else f"dec_{axis}.f32.bin"
        out = os.path.join(out_dir, fname)
        try:
            if os.path.isfile(out):
                os.remove(out)
        except OSError:
            pass

        cmd = [
            pressio_bin,
            "-i",
            inp,
            "-d",
            str(int(nt)),
            "-d",
            str(int(na)),
            "-t",
            "float",
            "-b",
            f"compressor={compressor}",
            "-o",
            f"rel={float(rel)}",
        ]
        for opt in pressio_opts:
            cmd.extend(["-o", opt])
        cmd.extend(["-W", out])

        # Avoid capturing output; pressio can be chatty.
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            raise RuntimeError(f"pressio failed (axis={axis}) exit_code={result.returncode}")
        if not os.path.isfile(out):
            raise FileNotFoundError(f"pressio -W did not produce output: {out}")
        got = os.path.getsize(out)
        if got != expected_bytes:
            try:
                os.remove(out)
            except OSError:
                pass
            raise RuntimeError(
                f"pressio -W size mismatch for axis={axis}: got {got} bytes, need {expected_bytes} (nt*na*4)"
            )
        dec_paths[axis] = out

    return dec_paths["x"], dec_paths["y"], dec_paths["z"]


def build_positions_from_xyz(x_path, y_path, z_path, nt, na, frame):
    x = np.fromfile(x_path, dtype=np.float32).reshape(nt, na)[frame]
    y = np.fromfile(y_path, dtype=np.float32).reshape(nt, na)[frame]
    z = np.fromfile(z_path, dtype=np.float32).reshape(nt, na)[frame]
    return np.stack([x, y, z], axis=1).astype(np.float32)


def pair_distance_axis(pos, bins):
    pos = np.asarray(pos, dtype=np.float64)
    shifted = pos - pos.min(axis=0)
    L = shifted.max(axis=0)
    rmax = float(np.min(L)) / 2.0
    edges = np.linspace(0.0, rmax, bins + 1, dtype=np.float64)
    r = (edges[:-1] + edges[1:]) / 2.0
    return r


def axis_from_rmax(rmax, bins):
    if rmax is None:
        return np.arange(bins, dtype=np.float64)
    edges = np.linspace(0.0, float(rmax), bins + 1, dtype=np.float64)
    return (edges[:-1] + edges[1:]) / 2.0


def resample_to_common_edges(r1, g1, r2, g2, bins=None, rmax_cap: Optional[float] = None):
    """Resample two RDF curves onto a shared pair-distance axis."""
    r1 = np.asarray(r1, dtype=np.float64)
    r2 = np.asarray(r2, dtype=np.float64)
    g1 = np.asarray(g1, dtype=np.float64)
    g2 = np.asarray(g2, dtype=np.float64)
    if r1.ndim != 1 or r2.ndim != 1 or g1.ndim != 1 or g2.ndim != 1:
        raise ValueError("r and g arrays must be 1D")
    if r1.size != g1.size or r2.size != g2.size:
        raise ValueError("Length mismatch between r and g arrays")
    if r1.size < 2 or r2.size < 2:
        raise ValueError("Need at least 2 bins to build common edges")

    common_rmax = min(float(r1[-1]), float(r2[-1]))
    if rmax_cap is not None and np.isfinite(float(rmax_cap)):
        common_rmax = min(common_rmax, float(rmax_cap))
    if bins is None:
        bins = min(r1.size, r2.size)
    if bins < 2:
        raise ValueError("bins must be >= 2")

    common_edges = np.linspace(0.0, common_rmax, int(bins) + 1, dtype=np.float64)
    common_r = (common_edges[:-1] + common_edges[1:]) / 2.0
    g1_common = np.interp(common_r, r1, g1)
    g2_common = np.interp(common_r, r2, g2)
    return common_r, g1_common, g2_common


def save_quad_pair_overlay_png(
    *,
    r_raw: np.ndarray,
    g_raw: np.ndarray,
    r_dec: np.ndarray,
    g_dec: np.ndarray,
    out_dir: str,
    dataset_name: str,
    compressor_name: str,
    error_bound: str,
    psnr_mean: Optional[float],
    compression_ratio: Optional[float],
    bit_rate: Optional[float],
    wasserstein: Optional[float],
    pair_distance_xmax: float,
    bins: int,
) -> str:
    """Single figure: original vs decompressed RDF. PSNR/CR/bit_rate match plot_rdf.py STANDARD aggregate."""
    cr_for_name = compression_ratio
    common_r, g0, g1 = resample_to_common_edges(
        r_raw, g_raw, r_dec, g_dec, bins=bins, rmax_cap=pair_distance_xmax
    )

    comp_f = _safe_filename_piece(compressor_name)
    eb_f = _safe_filename_piece(_EB_NORM_TO_CANON.get(_norm_eb(error_bound), _norm_eb(error_bound)))
    ds_f = _safe_filename_piece(dataset_name)
    psnr_piece = (
        f"{psnr_mean:.6g}" if psnr_mean is not None and np.isfinite(psnr_mean) else "PSNRNA"
    )
    cr_piece = f"{cr_for_name:.6g}" if cr_for_name is not None and np.isfinite(cr_for_name) else "CRNA"
    w_piece = (
        f"{wasserstein:.6g}" if wasserstein is not None and np.isfinite(wasserstein) else "WNA"
    )

    fname = f"rdf_overlay_{ds_f}_{comp_f}_{eb_f}_PSNR{psnr_piece}_CR{cr_piece}_W{w_piece}.png"
    path_out = os.path.join(out_dir, fname)

    plt.figure(figsize=(7, 5))
    plt.plot(common_r, g0, label="original (raw)", color="tab:blue", linewidth=1.8)
    plt.plot(
        common_r,
        g1,
        label=f"decompressed ({compressor_name}, eb={error_bound})",
        color="tab:orange",
        linewidth=1.65,
        alpha=0.9,
    )
    plt.xlabel("pair distance", fontsize=RDF_AXIS_LABEL_FONTSIZE)
    plt.ylabel("g(r)", fontsize=RDF_AXIS_LABEL_FONTSIZE)
    plt.xlim(0.0, float(pair_distance_xmax))
    _apply_rdf_axis_style()
    plt.legend(prop={"size": RDF_TICK_LABEL_FONTSIZE})
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path_out, dpi=180)
    plt.close()
    return path_out


def save_quad_original_only_png(
    *,
    r_raw: np.ndarray,
    g_raw: np.ndarray,
    out_dir: str,
    dataset_name: str,
    pair_distance_xmax: float,
) -> str:
    """Single curve: raw RDF only (same r,g as quad overlays' original). Written once per quad run."""
    ds_f = _safe_filename_piece(dataset_name)
    fname = f"rdf_original_only_{ds_f}.png"
    path_out = os.path.join(out_dir, fname)
    plt.figure(figsize=(7, 5))
    plt.plot(np.asarray(r_raw), np.asarray(g_raw), color="tab:blue", linewidth=1.8)
    plt.xlabel("pair distance", fontsize=RDF_AXIS_LABEL_FONTSIZE)
    plt.ylabel("g(r)", fontsize=RDF_AXIS_LABEL_FONTSIZE)
    plt.xlim(0.0, float(pair_distance_xmax))
    _apply_rdf_axis_style()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path_out, dpi=180)
    plt.close()
    return path_out


def plot_and_save(
    r1,
    g1,
    r2,
    g2,
    out_dir,
    *,
    compressor_name: str = "NA",
    error_bound: str = "NA",
    dataset_name: str = "NA",
    cr_total: Optional[float] = None,
    wasserstein_override: Optional[float] = None,
    psnr_mean: Optional[float] = None,
    overlay_only: bool = False,
    pair_distance_xmax: float = DEFAULT_PAIR_DISTANCE_XMAX,
):
    comp_piece = _safe_filename_piece(compressor_name)
    eb_piece = _safe_filename_piece(_EB_NORM_TO_CANON.get(_norm_eb(error_bound), _norm_eb(error_bound)))
    dataset_piece = _safe_filename_piece(dataset_name)
    cr_piece = (
        f"{cr_total:.6g}" if cr_total is not None and np.isfinite(cr_total) else "CRNA"
    )
    psnr_piece = (
        f"{psnr_mean:.6g}"
        if psnr_mean is not None and np.isfinite(psnr_mean)
        else "PSNRNA"
    )

    prefix = f"{comp_piece}_{eb_piece}_{dataset_piece}"

    if not overlay_only:
        np.save(os.path.join(out_dir, f"g1_rdf_raw_{prefix}.npy"), g1)
        np.save(os.path.join(out_dir, f"g2_rdf_dec_{prefix}.npy"), g2)
        np.save(os.path.join(out_dir, f"r_g1_pair_distance_{prefix}.npy"), r1)
        np.save(os.path.join(out_dir, f"r_g2_pair_distance_{prefix}.npy"), r2)

    # Overlay always uses a shared x-axis to make point-wise comparison fair.
    common_r, g1_common, g2_common = resample_to_common_edges(
        r1, g1, r2, g2, rmax_cap=pair_distance_xmax
    )

    def wasserstein_1d_discrete(x, w1, w2):
        """
        1D Wasserstein distance (EMD) for discrete distributions on a shared, sorted support x.
        Computes integral of |CDF1 - CDF2| dx.
        """
        x = np.asarray(x, dtype=np.float64)
        w1 = np.asarray(w1, dtype=np.float64)
        w2 = np.asarray(w2, dtype=np.float64)
        if x.ndim != 1 or w1.ndim != 1 or w2.ndim != 1:
            return np.nan
        if x.size < 2 or w1.size != x.size or w2.size != x.size:
            return np.nan
        w1 = np.maximum(w1, 0.0)
        w2 = np.maximum(w2, 0.0)
        s1 = float(np.sum(w1))
        s2 = float(np.sum(w2))
        if not np.isfinite(s1) or not np.isfinite(s2) or s1 <= 0.0 or s2 <= 0.0:
            return np.nan
        p1 = w1 / s1
        p2 = w2 / s2
        c1 = np.cumsum(p1)
        c2 = np.cumsum(p2)
        dx = np.diff(x)
        if not np.all(np.isfinite(dx)) or np.any(dx <= 0):
            return np.nan
        return float(np.sum(np.abs(c1[:-1] - c2[:-1]) * dx))

    if wasserstein_override is not None and np.isfinite(float(wasserstein_override)):
        w_dist = float(wasserstein_override)
    else:
        w_dist = wasserstein_1d_discrete(common_r, g1_common, g2_common)
    w_piece = f"{w_dist:.6g}" if np.isfinite(w_dist) else "WNA"

    if not overlay_only:
        plt.figure(figsize=(7, 5))
        plt.plot(r1, g1, color="tab:blue", linewidth=1.8)
        plt.xlabel("pair distance")
        plt.ylabel("gr")
        plt.xlim(0.0, float(pair_distance_xmax))
        plt.title(f"g1 (raw) [{prefix}]")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f"g1_raw_rdf_{prefix}_PSNR{psnr_piece}_W{w_piece}.png"),
            dpi=180,
        )
        plt.close()

        plt.figure(figsize=(7, 5))
        plt.plot(r2, g2, color="tab:orange", linewidth=1.8)
        plt.xlabel("pair distance")
        plt.ylabel("gr")
        plt.xlim(0.0, float(pair_distance_xmax))
        plt.title(f"g2 (decompressed) [{prefix}]")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f"g2_dec_rdf_{prefix}_PSNR{psnr_piece}_W{w_piece}.png"),
            dpi=180,
        )
        plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(common_r, g1_common, label="g1 raw", color="tab:blue", linewidth=1.6)
    plt.plot(
        common_r,
        g2_common,
        label="g2 decompressed",
        color="tab:orange",
        linewidth=1.6,
        alpha=0.85,
    )
    plt.xlabel("pair distance")
    plt.ylabel("gr")
    plt.xlim(0.0, float(pair_distance_xmax))
    plt.title(f"RDF comparison (g1 vs g2) CR={cr_piece} PSNR={psnr_piece} W={w_piece}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            out_dir,
            f"g1_g2_overlay_rdf_{prefix}_CR{cr_piece}_PSNR{psnr_piece}_W{w_piece}.png",
        ),
        dpi=180,
    )
    plt.close()

    return prefix, cr_piece, w_piece, psnr_piece


def _parse_quad_specs_colon(specs_str: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for part in str(specs_str).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid --quad_specs entry {part!r}; use compressor:error_bound")
        c, eb = part.split(":", 1)
        c, eb = c.strip(), eb.strip()
        if not c or not eb:
            raise ValueError(f"Invalid --quad_specs entry {part!r}")
        out.append((c, eb))
    if len(out) != 4:
        raise ValueError(
            f"--quad_specs must expand to exactly 4 compressor:error_bound pairs, got {len(out)}"
        )
    return out


def _resolve_quad_specs(
    specs_str: Optional[str],
    compressors_str: Optional[str],
    error_bounds_str: Optional[str],
) -> List[Tuple[str, str]]:
    """
    Quad mode compressor/bounds from CLI.
    Priority: --quad_specs (if non-empty) > --quad_compressors + --quad_error_bounds > QUAD_RDF_SPECS.
    """
    if specs_str and str(specs_str).strip():
        return _parse_quad_specs_colon(specs_str)

    has_c = compressors_str and str(compressors_str).strip()
    has_e = error_bounds_str and str(error_bounds_str).strip()
    if has_c or has_e:
        if not (has_c and has_e):
            raise ValueError(
                "With --quad, pass both --quad_compressors and --quad_error_bounds (four values each), "
                "or use --quad_specs."
            )
        comps = [x.strip() for x in str(compressors_str).split(",") if x.strip()]
        ebs = [x.strip() for x in str(error_bounds_str).split(",") if x.strip()]
        if len(comps) != 4 or len(ebs) != 4:
            raise ValueError(
                f"--quad_compressors and --quad_error_bounds must each list exactly 4 comma-separated "
                f"values (got {len(comps)} compressors and {len(ebs)} error bounds)."
            )
        return list(zip(comps, ebs))

    return list(QUAD_RDF_SPECS)


def run_quad_overlay_compute(
    *,
    specs: List[Tuple[str, str]],
    raw_prefix: str,
    nt: int,
    na: int,
    frame: int,
    bins: int,
    sample_ratio: Optional[float],
    pressio_bin: str,
    pressio_opts: list[str],
    work_dir: str,
    out_dir: str,
    dataset_name: str,
    pair_distance_xmax: float,
    exaalt_suite: str = "copper",
):
    from run_pressio_rdf import read_raw_frame, compute_rdf

    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    print(f"RDF plots: pair distance x-axis 0 .. {pair_distance_xmax} (set via --pair_distance_xmax)")

    coords_raw = read_raw_frame(raw_prefix, frame, nt, na)
    g_raw = compute_rdf(coords_raw, bins=bins, sample_ratio=sample_ratio)
    r_raw = pair_distance_axis(coords_raw, bins)

    saved_paths: List[str] = []
    for compressor_name, error_bound in specs:
        try:
            rel = float(error_bound)
        except ValueError as e:
            raise ValueError(f"error_bound must be numeric for pressio: {error_bound!r}") from e
        tag = _safe_filename_piece(f"{compressor_name}_{_norm_eb(error_bound)}")
        dec_x, dec_y, dec_z = pressio_decompress_xyz(
            pressio_bin=pressio_bin,
            raw_prefix=raw_prefix,
            nt=nt,
            na=na,
            compressor=compressor_name,
            rel=rel,
            pressio_opts=pressio_opts,
            out_dir=work_dir,
            out_tag=tag,
        )
        coords_dec = build_positions_from_xyz(dec_x, dec_y, dec_z, nt, na, frame)
        g_dec = compute_rdf(coords_dec, bins=bins, sample_ratio=sample_ratio)
        r_dec = pair_distance_axis(coords_dec, bins)

        agg = aggregate_standard_xyz_metrics_like_plot_rdf(
            compressor_name, error_bound, dataset_name, exaalt_suite=exaalt_suite
        )
        psnr = None
        cr_agg = None
        br_agg = None
        if agg:
            p = agg.get("psnr")
            if p is not None and np.isfinite(float(p)):
                psnr = float(p)
            c = agg.get("compression_ratio")
            if c is not None and np.isfinite(float(c)):
                cr_agg = float(c)
            b = agg.get("bit_rate")
            if b is not None and np.isfinite(float(b)):
                br_agg = float(b)
        w = lookup_wasserstein_from_rdf_csv(compressor_name, error_bound, dataset_name)

        path_png = save_quad_pair_overlay_png(
            r_raw=r_raw,
            g_raw=g_raw,
            r_dec=r_dec,
            g_dec=g_dec,
            out_dir=out_dir,
            dataset_name=dataset_name,
            compressor_name=compressor_name,
            error_bound=error_bound,
            psnr_mean=psnr,
            compression_ratio=cr_agg,
            bit_rate=br_agg,
            wasserstein=w,
            pair_distance_xmax=pair_distance_xmax,
            bins=bins,
        )
        saved_paths.append(path_png)
        print(f"Saved: {path_png}")

    orig_path = save_quad_original_only_png(
        r_raw=r_raw,
        g_raw=g_raw,
        out_dir=out_dir,
        dataset_name=dataset_name,
        pair_distance_xmax=pair_distance_xmax,
    )
    saved_paths.append(orig_path)
    print(f"Saved: {orig_path}")

    return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description="Plot RDF g1(raw) and g2(decompressed) with pair distance axis."
    )
    parser.add_argument("--from_npy", action="store_true", help="Plot directly from existing .npy RDF files")
    parser.add_argument("--rdf_raw_npy", default="rdf_raw.npy", help="Path to raw RDF .npy")
    parser.add_argument("--rdf_dec_npy", default="rdf_dec.npy", help="Path to decompressed RDF .npy")
    parser.add_argument("--r_raw_npy", default=None, help="Optional pair-distance axis .npy for raw RDF")
    parser.add_argument("--r_dec_npy", default=None, help="Optional pair-distance axis .npy for decompressed RDF")
    parser.add_argument("--rmax_raw", type=float, default=None, help="r_max for raw RDF (if no r_raw_npy)")
    parser.add_argument("--rmax_dec", type=float, default=None, help="r_max for dec RDF (if no r_dec_npy)")
    parser.add_argument("--decompressed_x", default=None, help="Decompressed x file (compute mode)")
    parser.add_argument("--decompressed_y", default=None, help="Decompressed y file (compute mode)")
    parser.add_argument("--decompressed_z", default=None, help="Decompressed z file (compute mode)")
    parser.add_argument("--raw_prefix", default=None, help="Raw .x/.y/.z.f32.dat prefix (compute mode)")
    parser.add_argument("--nt", type=int, default=None, help="Number of timesteps (compute mode)")
    parser.add_argument("--na", type=int, default=None, help="Number of atoms (compute mode)")
    parser.add_argument("--frame", type=int, default=None, help="Frame index (compute mode)")
    parser.add_argument("--bins", type=int, default=300, help="Number of RDF bins (compute mode)")
    parser.add_argument("--sample_ratio", type=float, default=None, help="Sampling ratio for compute_rdf")
    parser.add_argument(
        "--run_pressio",
        action="store_true",
        help="In compute mode, run pressio to compress+decompress x/y/z automatically (writes dec files to out_dir).",
    )
    parser.add_argument(
        "--pressio",
        default="pressio",
        help="Pressio CLI path/name (default: pressio). Used with --run_pressio.",
    )
    parser.add_argument(
        "--pressio_opts",
        action="append",
        default=[],
        help="Extra pressio options as key=value, repeatable (e.g. sz3:algorithm_str=ALGO_LORENZO_REG). Used with --run_pressio.",
    )
    parser.add_argument("--compressor_name", default="NA", help="Compressor name (for naming and CR)")
    parser.add_argument("--error_bound", default="NA", help="Error bound (for naming and CR)")
    parser.add_argument("--dataset_name", default="NA", help="Dataset name (for naming and CR)")
    parser.add_argument(
        "--exaalt_suite",
        choices=("copper", "helium"),
        default="copper",
        help="STANDARD/EXAALT folder prefix SDRBENCH-exaalt-{copper|helium}_… for PSNR/CR metrics (default: copper).",
    )
    parser.add_argument(
        "--rdf_csv",
        default=None,
        help=(
            "Optional path to a *_{compressor}_rdf.csv under compression_framework/outputs/RDF "
            "(overrides auto-discovery for Wasserstein in plot title/filename)."
        ),
    )
    parser.add_argument(
        "--out_dir",
        default=".",
        help="Output directory for png and npy files (default: current directory)",
    )
    parser.add_argument(
        "--overlay_only",
        action="store_true",
        help="Only write the g1 vs g2 overlay PNG (no separate g1/g2 figures, no .npy dumps).",
    )
    parser.add_argument(
        "--pair_distance_xmax",
        type=float,
        default=DEFAULT_PAIR_DISTANCE_XMAX,
        help="Plot xlim upper bound for pair distance (default: 8).",
    )
    parser.add_argument(
        "--quad",
        action="store_true",
        help="Four separate overlays (raw vs one decompressor each). Set compressors/EB via --quad_specs or --quad_compressors + --quad_error_bounds. Writes four PNGs plus rdf_original_only_<dataset>.png to --out_dir; overlay names include PSNR + compression_ratio.",
    )
    parser.add_argument(
        "--quad_specs",
        default=None,
        help='Optional: four compressor:error_bound pairs, comma-separated, e.g. zfp:1e-2,mgard:1e-3,sz3:1e-2,sperr:5e-2. If set, overrides --quad_compressors/--quad_error_bounds.',
    )
    parser.add_argument(
        "--quad_compressors",
        default=None,
        help="With --quad: four comma-separated names, e.g. zfp,mgard,sz3,sperr (pair with --quad_error_bounds).",
    )
    parser.add_argument(
        "--quad_error_bounds",
        default=None,
        help="With --quad: four comma-separated relative errors matching --quad_compressors order, e.g. 1e-2,1e-3,5e-3,1e-2.",
    )
    parser.add_argument(
        "--quad_work_dir",
        default=None,
        help="Directory for temporary pressio dec outputs in --quad mode (default: <out_dir>/_quad_pressio_tmp).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print debug RDF bin diagnostics.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.quad:
        if args.from_npy:
            raise ValueError("--quad requires compute mode (omit --from_npy).")
        from run_pressio_rdf import DEFAULT_NA, DEFAULT_NT, DEFAULT_RAW_PREFIX, timestep

        specs = _resolve_quad_specs(
            args.quad_specs, args.quad_compressors, args.quad_error_bounds
        )
        nt = DEFAULT_NT if args.nt is None else args.nt
        na = DEFAULT_NA if args.na is None else args.na
        frame = timestep if args.frame is None else args.frame
        raw_prefix = DEFAULT_RAW_PREFIX if args.raw_prefix is None else args.raw_prefix
        work_dir = args.quad_work_dir or os.path.join(args.out_dir, "_quad_pressio_tmp")
        if args.dataset_name in ("", "NA"):
            raise ValueError("--quad requires --dataset_name (e.g. dataset2-83x1077290 for CR/PSNR/W labels).")

        run_quad_overlay_compute(
            specs=specs,
            raw_prefix=raw_prefix,
            nt=nt,
            na=na,
            frame=frame,
            bins=args.bins,
            sample_ratio=args.sample_ratio,
            pressio_bin=args.pressio,
            pressio_opts=args.pressio_opts,
            work_dir=work_dir,
            out_dir=args.out_dir,
            dataset_name=args.dataset_name,
            pair_distance_xmax=float(args.pair_distance_xmax),
            exaalt_suite=args.exaalt_suite,
        )
        return

    if args.from_npy:
        # Auto-detect when user didn't pass explicit paths (keeps backward compatibility).
        raw_is_default = args.rdf_raw_npy == "rdf_raw.npy"
        dec_is_default = args.rdf_dec_npy == "rdf_dec.npy"
        if raw_is_default and dec_is_default:
            args.rdf_raw_npy, args.rdf_dec_npy = auto_find_rdf_npy_paths(args.out_dir)

        if not os.path.isfile(args.rdf_raw_npy):
            raise FileNotFoundError(f"Missing file: {args.rdf_raw_npy}")
        if not os.path.isfile(args.rdf_dec_npy):
            raise FileNotFoundError(f"Missing file: {args.rdf_dec_npy}")

        g1 = np.load(args.rdf_raw_npy)
        g2 = np.load(args.rdf_dec_npy)

        if g1.ndim != 1 or g2.ndim != 1:
            raise ValueError("rdf_raw.npy and rdf_dec.npy must be 1D arrays")

        if args.r_raw_npy is not None:
            if not os.path.isfile(args.r_raw_npy):
                raise FileNotFoundError(f"Missing file: {args.r_raw_npy}")
            r1 = np.load(args.r_raw_npy)
        else:
            r1 = axis_from_rmax(args.rmax_raw, g1.shape[0])

        if args.r_dec_npy is not None:
            if not os.path.isfile(args.r_dec_npy):
                raise FileNotFoundError(f"Missing file: {args.r_dec_npy}")
            r2 = np.load(args.r_dec_npy)
        else:
            r2 = axis_from_rmax(args.rmax_dec, g2.shape[0])

        if r1.shape[0] != g1.shape[0]:
            raise ValueError("Length mismatch: r1 vs g1")
        if r2.shape[0] != g2.shape[0]:
            raise ValueError("Length mismatch: r2 vs g2")

        coords_raw = coords_dec = None

    else:
        from run_pressio_rdf import (
            DEFAULT_NA,
            DEFAULT_NT,
            DEFAULT_RAW_PREFIX,
            timestep,
            read_raw_frame,
            compute_rdf,
        )

        nt = DEFAULT_NT if args.nt is None else args.nt
        na = DEFAULT_NA if args.na is None else args.na
        frame = timestep if args.frame is None else args.frame
        raw_prefix = DEFAULT_RAW_PREFIX if args.raw_prefix is None else args.raw_prefix

        if args.run_pressio:
            if args.compressor_name in ("", "NA"):
                raise ValueError("--run_pressio requires --compressor_name (e.g. sz3/zfp/sperr/mgard)")
            if args.error_bound in ("", "NA"):
                raise ValueError("--run_pressio requires --error_bound (used as pressio rel)")
            try:
                rel = float(args.error_bound)
            except ValueError as e:
                raise ValueError(f"--error_bound must be a float when using --run_pressio (got {args.error_bound!r})") from e

            dec_x, dec_y, dec_z = pressio_decompress_xyz(
                pressio_bin=args.pressio,
                raw_prefix=raw_prefix,
                nt=nt,
                na=na,
                compressor=args.compressor_name,
                rel=rel,
                pressio_opts=args.pressio_opts,
                out_dir=args.out_dir,
            )
            args.decompressed_x, args.decompressed_y, args.decompressed_z = dec_x, dec_y, dec_z
        else:
            for path in (args.decompressed_x, args.decompressed_y, args.decompressed_z):
                if path is None or not os.path.isfile(path):
                    raise FileNotFoundError(
                        "Compute mode requires valid --decompressed_x/--decompressed_y/--decompressed_z "
                        "(or pass --run_pressio to generate them automatically)."
                    )

        coords_raw = read_raw_frame(raw_prefix, frame, nt, na)
        coords_dec = build_positions_from_xyz(
            args.decompressed_x, args.decompressed_y, args.decompressed_z, nt, na, frame
        )
        g1 = compute_rdf(coords_raw, bins=args.bins, sample_ratio=args.sample_ratio)
        g2 = compute_rdf(coords_dec, bins=args.bins, sample_ratio=args.sample_ratio)
        r1 = pair_distance_axis(coords_raw, args.bins)
        r2 = pair_distance_axis(coords_dec, args.bins)

    if args.verbose:
        print(g1[:10])
        print(g2[:10])
        if not args.from_npy:
            uniq_raw = np.unique(coords_raw, axis=0)
            uniq_dec = np.unique(coords_dec, axis=0)
            print("raw duplicates:", len(coords_raw) - len(uniq_raw))
            print("dec duplicates:", len(coords_dec) - len(uniq_dec))
        nz = np.where(g2 > 0)[0]
        print("first nonzero bin of dec:", nz[0] if len(nz) else None)
        print("first few nonzero bins:", nz[:20])
        print("values:", g2[nz[:20]])
    cr_total = compute_cr_total_from_standard_csv(
        args.compressor_name,
        args.error_bound,
        args.dataset_name,
        exaalt_suite=args.exaalt_suite,
    )
    if cr_total is None:
        print(
            "Warning: could not compute CR from STANDARD csv; overlay filename will include CRNA."
        )
    else:
        print(f"CR computed from STANDARD sizes: {cr_total}")

    w_csv = lookup_wasserstein_from_rdf_csv(
        args.compressor_name,
        args.error_bound,
        args.dataset_name,
        explicit_csv=args.rdf_csv,
    )
    if w_csv is not None:
        print(f"Wasserstein from RDF outputs CSV: {w_csv}")
    else:
        print(
            "Warning: no wasserstein_distance in RDF CSV (check dataset/compressor/eb); "
            "using discrete 1D EMD on resampled g(r) for plot label."
        )

    psnr_mean = compute_mean_psnr_from_standard_csv(
        args.compressor_name,
        args.error_bound,
        args.dataset_name,
        exaalt_suite=args.exaalt_suite,
    )
    if psnr_mean is None:
        print("Warning: could not compute mean PSNR from STANDARD csv; filename will use PSNRNA.")
    else:
        print(f"Mean PSNR (x,y,z) from STANDARD: {psnr_mean}")

    prefix, cr_piece, w_piece, psnr_piece = plot_and_save(
        r1,
        g1,
        r2,
        g2,
        args.out_dir,
        compressor_name=args.compressor_name,
        error_bound=args.error_bound,
        dataset_name=args.dataset_name,
        cr_total=cr_total,
        wasserstein_override=w_csv,
        psnr_mean=psnr_mean,
        overlay_only=args.overlay_only,
        pair_distance_xmax=float(args.pair_distance_xmax),
    )

    print(f"Saved plots and arrays to: {os.path.abspath(args.out_dir)}")
    if not args.overlay_only:
        print(f" - g1_raw_rdf_{prefix}_PSNR{psnr_piece}_W{w_piece}.png")
        print(f" - g2_dec_rdf_{prefix}_PSNR{psnr_piece}_W{w_piece}.png")
    print(f" - g1_g2_overlay_rdf_{prefix}_CR{cr_piece}_PSNR{psnr_piece}_W{w_piece}.png")
    if not args.overlay_only:
        print(f" - g1_rdf_raw_{prefix}.npy, g2_rdf_dec_{prefix}.npy")
        print(f" - r_g1_pair_distance_{prefix}.npy, r_g2_pair_distance_{prefix}.npy")
    if args.from_npy and args.r_raw_npy is None and args.r_dec_npy is None:
        print("Note: no r-axis .npy provided; use --rmax_raw/--rmax_dec for true pair-distance scale.")


if __name__ == "__main__":
    main()
