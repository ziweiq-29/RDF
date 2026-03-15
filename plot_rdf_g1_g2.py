#!/usr/bin/env python3
"""Plot RDF curves (g1/g2) with x=pair distance and y=gr.

This script supports two modes:
1) --from_npy: read existing rdf_raw.npy / rdf_dec.npy directly
2) default: compute RDF from raw/decompressed coordinates, then plot
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


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


def plot_and_save(r1, g1, r2, g2, out_dir):
    np.save(os.path.join(out_dir, "g1_rdf_raw.npy"), g1)
    np.save(os.path.join(out_dir, "g2_rdf_dec.npy"), g2)
    np.save(os.path.join(out_dir, "r_g1_pair_distance.npy"), r1)
    np.save(os.path.join(out_dir, "r_g2_pair_distance.npy"), r2)

    plt.figure(figsize=(7, 5))
    plt.plot(r1, g1, color="tab:blue", linewidth=1.8)
    plt.xlabel("pair distance")
    plt.ylabel("gr")
    plt.title("g1 (raw)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "g1_raw_rdf.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(r2, g2, color="tab:orange", linewidth=1.8)
    plt.xlabel("pair distance")
    plt.ylabel("gr")
    plt.title("g2 (decompressed)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "g2_dec_rdf.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(r1, g1, label="g1 raw", color="tab:blue", linewidth=1.6)
    plt.plot(r2, g2, label="g2 decompressed", color="tab:orange", linewidth=1.6, alpha=0.85)
    plt.xlabel("pair distance")
    plt.ylabel("gr")
    plt.title("RDF comparison (g1 vs g2)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "g1_g2_overlay_rdf.png"), dpi=180)
    plt.close()


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
        "--out_dir",
        default=".",
        help="Output directory for png and npy files (default: current directory)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.from_npy:
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

        for path in (args.decompressed_x, args.decompressed_y, args.decompressed_z):
            if path is None or not os.path.isfile(path):
                raise FileNotFoundError(
                    "Compute mode requires valid --decompressed_x/--decompressed_y/--decompressed_z"
                )

        coords_raw = read_raw_frame(raw_prefix, frame, nt, na)
        coords_dec = build_positions_from_xyz(
            args.decompressed_x, args.decompressed_y, args.decompressed_z, nt, na, frame
        )
        g1 = compute_rdf(coords_raw, bins=args.bins, sample_ratio=args.sample_ratio)
        g2 = compute_rdf(coords_dec, bins=args.bins, sample_ratio=args.sample_ratio)
        r1 = pair_distance_axis(coords_raw, args.bins)
        r2 = pair_distance_axis(coords_dec, args.bins)
    print(g1[:10])
    print(g2[:10])
    plot_and_save(r1, g1, r2, g2, args.out_dir)

    print(f"Saved plots and arrays to: {os.path.abspath(args.out_dir)}")
    print(" - g1_raw_rdf.png")
    print(" - g2_dec_rdf.png")
    print(" - g1_g2_overlay_rdf.png")
    print(" - g1_rdf_raw.npy, g2_rdf_dec.npy")
    print(" - r_g1_pair_distance.npy, r_g2_pair_distance.npy")
    if args.from_npy and args.r_raw_npy is None and args.r_dec_npy is None:
        print("Note: no r-axis .npy provided; use --rmax_raw/--rmax_dec for true pair-distance scale.")


if __name__ == "__main__":
    main()
