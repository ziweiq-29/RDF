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
import shutil
import subprocess
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RDF_SCRIPT = os.path.join(SCRIPT_DIR, "run_pressio_rdf.py")


def output_metrics(metrics):
    print("external:api=json:1", flush=True)
    print(json.dumps(metrics), flush=True)


parser = argparse.ArgumentParser()
parser.add_argument("--standalone", action="store_true",
                    help="Invoked by run_pressio_external; run pressio 3× (x,y,z) inside pipeline, then RDF, then output metrics")
parser.add_argument("--external_mode", action="store_true")
parser.add_argument("--axis", choices=("x", "y", "z"), help="Current axis (LibPressio injects; only with --external_mode)")
parser.add_argument("--input", help="Input path (LibPressio injects temp original)")
parser.add_argument("--decompressed", help="Decompressed path (LibPressio injects; use for current axis)")
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
# STANDALONE — 由 run_pressio_external.py 调用：仅在此处运行 pressio（3× x,y,z），再 RDF，再输出
# ============================================================
if args.standalone:
    prefix = args.original_input or ""
    if not prefix:
        output_metrics({"dists": []})
        sys.exit(1)
    prefix = (
        prefix.replace(".x.f32.dat", "")
        .replace(".y.f32.dat", "")
        .replace(".z.f32.dat", "")
        .rstrip(".")
    )
    for axis in "xyz":
        if not os.path.isfile(f"{prefix}.{axis}.f32.dat"):
            sys.stderr.write(f"Missing {prefix}.{axis}.f32.dat\n")
            output_metrics({"dists": []})
            sys.exit(1)
    dec_dir = os.path.join(SCRIPT_DIR, "rdf_tmp")
    os.makedirs(dec_dir, exist_ok=True)
    dec_paths = {}
    axes_to_run_pressio = list("xyz")
    expected_bytes = nt * na * 4
    # STEP 1 — pressio -W for x, y, z
    for axis in axes_to_run_pressio:
        inp = f"{prefix}.{axis}.f32.dat"
        out = os.path.join(dec_dir, f"{axis}.bin")
        # 必须删掉旧文件再 -W，否则若 pressio 只写部分或失败，会留下 100x8192000 等残留 → reshape 错
        try:
            if os.path.isfile(out):
                os.remove(out)
        except OSError:
            pass
        cmd = [
            args.pressio,
            "-i", inp,
            "-d", str(nt), "-d", str(na),
            "-t", "float",
            "-b", f"compressor={args.compressor}",
            "-o", f"rel={args.rel}",
        ]
        for opt in args.pressio_opts:
            cmd.extend(["-o", opt])
        cmd.extend(["-W", out])
        if args.print_pressio_cmd:
            sys.stderr.write("pressio: " + " ".join(cmd) + "\n")
            sys.exit(0)
        # 禁止 capture_output：pressio 会刷大量 composite 行，PIPE 塞满 → 死锁，外层只看到 returncode!=0 且日志空
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            sys.stderr.write(f"[pipeline] axis={axis} pressio -W exit {result.returncode}\n")
            sys.stderr.flush()
            output_metrics({"dists": []})
            sys.exit(1)
        if not os.path.isfile(out):
            sys.stderr.write(f"[pipeline] pressio -W missing output {out}\n")
            output_metrics({"dists": []})
            sys.exit(1)
        got = os.path.getsize(out)
        if got != expected_bytes:
            sys.stderr.write(
                f"[pipeline] axis={axis} -W size mismatch: got {got} bytes, "
                f"need {expected_bytes} (nt*na*4={nt}*{na}*4). Remove stale rdf_tmp/*.bin and retry.\n"
            )
            sys.stderr.flush()
            try:
                os.remove(out)
            except OSError:
                pass
            output_metrics({"dists": []})
            sys.exit(1)
        dec_paths[axis] = out
    # STEP 2 — RDF script
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
    # STEP 3 — output metrics
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

# ============================================================
# PROBE — 无 standalone 且无 external_mode 时直接返回空
# ============================================================
if not args.external_mode:
    output_metrics({"dists": []})
    sys.exit(0)

# ============================================================
# TOP-LEVEL EXTERNAL MODE — use LibPressio's --decompressed for current axis, run pressio only for the other two
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

# LibPressio 未传 --decompressed/--axis 时（如 probe）直接返回空，不再跑 pressio
if not args.decompressed or not args.axis:
    output_metrics({"dists": []})
    sys.exit(0)

# Ensure raw files exist
for axis in "xyz":
    if not os.path.isfile(f"{prefix}.{axis}.f32.dat"):
        output_metrics({"dists": []})
        sys.exit(0)

# When LibPressio passes --decompressed and --axis, use that file for that axis.
# 只对「还没有 .bin 的轴」再跑 pressio；已有 .bin 的轴（本次 --decompressed 或之前某次调用写的）直接复用，避免同一轴被压缩多次。
dec_dir = os.path.join(SCRIPT_DIR, "rdf_tmp")
os.makedirs(dec_dir, exist_ok=True)
dec_paths = {}

if args.decompressed and args.axis and os.path.isfile(args.decompressed):
    dest = os.path.join(dec_dir, f"{args.axis}.bin")
    shutil.copy2(args.decompressed, dest)
    dec_paths[args.axis] = dest

# 复用之前某次 external 调用已写好的 .bin（run_pressio_external 对 x/y/z 各调一次，先调到的轴会先写入）
for a in "xyz":
    if a not in dec_paths:
        p = os.path.join(dec_dir, f"{a}.bin")
        if os.path.isfile(p):
            dec_paths[a] = p

axes_to_run_pressio = [a for a in "xyz" if a not in dec_paths]

# STEP 1 — run pressio -W only for axes not provided by LibPressio
expected_bytes = nt * na * 4
for axis in axes_to_run_pressio:
    inp = f"{prefix}.{axis}.f32.dat"
    out = os.path.join(dec_dir, f"{axis}.bin")
    # 与 standalone 一致：先删再 -W，避免残留；写后验长，否则 reshape 在 run_pressio_rdf 里才崩且看不到本消息
    try:
        if os.path.isfile(out):
            os.remove(out)
    except OSError:
        pass

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

    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if result.returncode != 0:
        sys.stderr.write(f"[pipeline] axis={axis} pressio -W exit {result.returncode}\n")
        sys.stderr.flush()
        output_metrics({"dists": []})
        sys.exit(1)

    if not os.path.isfile(out):
        sys.stderr.write(f"[pipeline] axis={axis} pressio -W missing output {out}\n")
        output_metrics({"dists": []})
        sys.exit(1)
    got = os.path.getsize(out)
    if got != expected_bytes:
        msg = (
            f"[pipeline] axis={axis} -W size mismatch: got {got} bytes, "
            f"need {expected_bytes} (nt*na*4={nt}*{na}*4). "
            f"pressio -W truncated; use smaller nt*na or chunked pipeline.\n"
        )
        sys.stderr.write(msg)
        sys.stderr.flush()
        try:
            os.remove(out)
        except OSError:
            pass
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
# STEP 3 — write metrics to file, stdout only metrics_file（qoi.cc 从文件读 vector）
# ============================================================

dists_path = os.path.join(SCRIPT_DIR, "dists.npy")
rdf_raw_path = os.path.join(SCRIPT_DIR, "rdf_raw.npy")
rdf_dec_path = os.path.join(SCRIPT_DIR, "rdf_dec.npy")

if not os.path.isfile(dists_path):
    output_metrics({"dists": [], "mass_orig": [], "mass_dec": []})
    sys.exit(1)

metrics = {
    "dists": np.load(dists_path).tolist(),
    "mass_orig": np.load(rdf_raw_path).tolist(),
    "mass_dec": np.load(rdf_dec_path).tolist(),
}
rdf_metrics_file = os.path.join(SCRIPT_DIR, "rdf_metrics.json")
with open(rdf_metrics_file, "w") as f:
    json.dump(metrics, f)
print("external:api=json:1", flush=True)
print(json.dumps({"metrics_file": os.path.abspath(rdf_metrics_file)}), flush=True)
sys.exit(0)
