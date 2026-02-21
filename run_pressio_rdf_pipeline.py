#!/usr/bin/env python3
"""
Stable RDF pipeline for LibPressio external QOI.

Key design:
- NEVER read .pressiooutXXXX streaming buffers
- ALWAYS decompress via pressio -W to real files
- Then run standalone RDF script
"""

import argparse
import json
import os
import subprocess
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RDF_SCRIPT = os.path.join(SCRIPT_DIR, "run_pressio_rdf.py")


def output_metrics(metrics):
    print("external:api=json:1", flush=True)
    print(json.dumps(metrics), flush=True)


parser = argparse.ArgumentParser()
parser.add_argument("--external_mode", action="store_true")
parser.add_argument("--input", help="Input path (LibPressio may inject; we prefer original_input)")
parser.add_argument("--original_input", help="Original dataset prefix (e.g. /path/to/dataset1-5423x3137)")
parser.add_argument("--dim", type=int, action="append", help="Dimensions (e.g. --dim 5423 --dim 3137)")
parser.add_argument("--compressor", default="sperr", help="Compressor (default: sperr)")
parser.add_argument("--rel", type=float, default=1e-4, help="Relative error (default: 1e-4)")
parser.add_argument("--pressio", default="pressio", help="Pressio CLI (default: pressio)")
parser.add_argument("--pressio-opts", action="append", default=[], help="Extra pressio options as key=value (e.g. sz3:algorithm_str=ALGO_LORENZO_REG)")
parser.add_argument("--print-pressio-cmd", action="store_true", help="Print the pressio -W command for the first axis and exit (for debugging).")

args, _ = parser.parse_known_args()

dims = args.dim or []
if len(dims) < 2:
    output_metrics({"dists": []})
    sys.exit(0)

nt, na = dims[:2]

# ============================================================
# PROBE CALL (LibPressio calls once without decompressed)
# ============================================================
if not args.external_mode:
    output_metrics({"dists": []})
    sys.exit(0)

# ============================================================
# TOP-LEVEL EXTERNAL MODE — never use --decompressed
# ============================================================

prefix = args.original_input or args.input
if not prefix:
    output_metrics({"dists": []})
    sys.exit(0)

prefix = (
    prefix.replace(".x.f32.dat", "")
    .replace(".y.f32.dat", "")
    .replace(".z.f32.dat", "")
    .rstrip(".")
)

# Ensure raw files exist
for axis in "xyz":
    if not os.path.isfile(f"{prefix}.{axis}.f32.dat"):
        output_metrics({"dists": []})
        sys.exit(0)

# ============================================================
# STEP 1 — decompress using pressio -W (REAL FILES)
# ============================================================

dec_dir = os.path.join(SCRIPT_DIR, "rdf_tmp")
os.makedirs(dec_dir, exist_ok=True)

dec_paths = {}

for axis in "xyz":
    inp = f"{prefix}.{axis}.f32.dat"
    out = os.path.join(dec_dir, f"{axis}.bin")

    cmd = [
        args.pressio,
        "-i", inp,
        "-d", str(nt),
        "-d", str(na),
        "-t", "float",
        "-b", f"compressor={args.compressor}",
        "-o", f"rel={args.rel}",
    ]
    for opt in args.pressio_opts:
        cmd.extend(["-o", opt])
    cmd.extend(["-W", out])

    if args.print_pressio_cmd:
        sys.stderr.write("pressio command: " + " ".join(cmd) + "\n")
        sys.exit(0)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        sys.stderr.write(result.stderr or "")
        output_metrics({"dists": []})
        sys.exit(1)

    dec_paths[axis] = out

# ============================================================
# STEP 2 — run standalone RDF script
# ============================================================

rdf_cmd = [
    sys.executable,
    RDF_SCRIPT,
    "--standalone",
    "--raw_prefix", prefix,
    "--nt", str(nt),
    "--na", str(na),
    "--decompressed_x", dec_paths["x"],
    "--decompressed_y", dec_paths["y"],
    "--decompressed_z", dec_paths["z"],
]

res = subprocess.run(rdf_cmd, capture_output=True, text=True, cwd=SCRIPT_DIR)

if res.returncode != 0:
    sys.stderr.write(res.stderr or "")
    output_metrics({"dists": []})
    sys.exit(1)

# ============================================================
# STEP 3 — read RDF outputs
# ============================================================

dists_path = os.path.join(SCRIPT_DIR, "dists.npy")
rdf_raw_path = os.path.join(SCRIPT_DIR, "rdf_raw.npy")
rdf_dec_path = os.path.join(SCRIPT_DIR, "rdf_dec.npy")

if not os.path.isfile(dists_path):
    output_metrics({"dists": []})
    sys.exit(1)

metrics = {
    "dists": np.load(dists_path).tolist(),
    "mass_orig": np.load(rdf_raw_path).tolist(),
    "mass_dec": np.load(rdf_dec_path).tolist(),
}

output_metrics(metrics)
sys.exit(0)
