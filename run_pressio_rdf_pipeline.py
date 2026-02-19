#!/usr/bin/env python3
"""
RDF pipeline: as external cmd → run_pressio_rdf.py, output LibPressio JSON;
as top-level (--external_mode) → run pressio -i 3 times (x,y,z), then RDF → dists.

LibPressio 只认这两行输出（qoi.cc 从 stdout 读）:
  print("external:api=json:1", file=sys.stdout, flush=True)
  print(json.dumps(metrics), file=sys.stdout, flush=True)

--- 用 pressio external:command 调用，qoi 才能拿到 dists（对 x/y/z 各跑一次）---
  PIPELINE="/path/to/run_pressio_rdf_pipeline.py"
  PREFIX="/path/to/dataset1-7852x1037"
  pressio -i "${PREFIX}.x.f32.dat" -d 7852 -d 1037 -t float \\
    -b compressor=zfp -o rel=5e-4 \\
    -b qoi:metric=external \\
    -o "external:command=python ${PIPELINE} --external_mode" \\
    -b external:launch_metric=print -o external:use_many=1 -m qoi -M all
  # 对 .y.f32.dat、.z.f32.dat 再各跑一次；第 3 次会输出 dists 给 qoi.cc
"""
import argparse
import json
import os
import subprocess
import sys
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))


def output_metrics(metrics):
    print("external:api=json:1", file=sys.stdout, flush=True)
    print(json.dumps(metrics), file=sys.stdout, flush=True)


# ------------ Command-line args (parse_known_args for LibPressio-injected args) ------------
parser = argparse.ArgumentParser(
    description="Pipeline: as external cmd delegates to run_pressio_rdf; as top-level runs pressio 3x then RDF"
)
parser.add_argument("--external_mode", action="store_true", help="Top-level: run pressio then read dists")
parser.add_argument("--input", help="Input path (prefix for top-level; per-file when external)")
parser.add_argument("--decompressed", help="Decompressed path (LibPressio injects when external)")
parser.add_argument("--original_input", help="Original input; pass in external:command")
parser.add_argument("--dim", type=int, action="append", help="Dimensions (e.g. --dim 7852 --dim 1037)")
parser.add_argument("--rel", type=float, default=1e-4, help="Relative error (default: 1e-4)")
parser.add_argument("--compressor", default="sperr", help="Compressor (default: sperr)")
parser.add_argument("--pressio", default="pressio", help="Pressio command (default: pressio)")
parser.add_argument("--rdf_script", default=None, help="Path to run_pressio_rdf.py (default: same dir)")
args, _ = parser.parse_known_args()

input_file = args.input
dims = args.dim or []
rel = args.rel
compressor = args.compressor
pressio = args.pressio
original_input = args.original_input or args.input
rdf_script = args.rdf_script or os.path.join(script_dir, "run_pressio_rdf.py")

# ------------ 被 LibPressio 作为 external 调用 → 转给 run_pressio_rdf，只输出 dists ------------
# 首次 launch（probe）无 --decompressed，输出默认格式避免 return 1
if not args.decompressed:
    output_metrics({"dists": []})
    sys.exit(0)

if args.decompressed:
    if not args.input or len(dims) < 2:
        output_metrics({"dists": []})
        sys.exit(0)
    nt, na = dims[0], dims[1]
    cache_dir = os.path.join(script_dir, "rdf_external_cache")
    os.makedirs(cache_dir, exist_ok=True)
    dec_x = os.path.join(cache_dir, "x.bin")
    dec_y = os.path.join(cache_dir, "y.bin")
    dec_z = os.path.join(cache_dir, "z.bin")
    # LibPressio 传的 --input 是临时文件路径，无法从路径判断 x/y/z；按 cache 顺序：先到为 x，次为 y，最后为 z
    if not os.path.isfile(dec_x):
        axis = "x"
    elif not os.path.isfile(dec_y):
        axis = "y"
    else:
        axis = "z"
    try:
        data = open(args.decompressed, "rb").read()
    except Exception:
        output_metrics({"dists": []})
        sys.exit(0)
    cache_bin = os.path.join(cache_dir, f"{axis}.bin")
    with open(cache_bin, "wb") as f:
        f.write(data)
    # raw_prefix 必须由 external:command 传入，例如 --original_input /path/to/dataset1-7852x1037
    raw_prefix = (args.original_input or "").strip()
    if not raw_prefix:
        raw_prefix = args.input.replace(".x.f32.dat", "").replace(".y.f32.dat", "").replace(".z.f32.dat", "").rstrip(".")
    if not raw_prefix or not os.path.isfile(f"{raw_prefix}.x.f32.dat"):
        output_metrics({"dists": []})
        sys.exit(0)
    if not all(os.path.isfile(p) for p in (dec_x, dec_y, dec_z)):
        output_metrics({"dists": []})
        sys.exit(0)
    if not os.path.isfile(rdf_script):
        output_metrics({"dists": []})
        sys.exit(1)
    rdf_cmd = [
        sys.executable,
        rdf_script,
        "--standalone",
        "--raw_prefix", raw_prefix,
        "--nt", str(nt),
        "--na", str(na),
        "--decompressed_x", dec_x,
        "--decompressed_y", dec_y,
        "--decompressed_z", dec_z,
    ]
    result = subprocess.run(rdf_cmd, capture_output=True, text=True)
    if result.stderr:
        sys.stderr.write(result.stderr)
    if result.returncode != 0:
        output_metrics({"dists": []})
        sys.exit(result.returncode)
    rdf_dir = os.path.dirname(rdf_script)
    dists_path = os.path.join(rdf_dir, "dists.npy")
    if not os.path.isfile(dists_path):
        output_metrics({"dists": []})
        sys.exit(1)
    dists = np.load(dists_path)
    # stdout 只打 dists；rdf_raw/rdf_dec 写入 script_dir 下的 rdf_curves.json
    rdf_raw = np.load( "rdf_raw.npy")
    rdf_dec= np.load( "rdf_dec.npy")
    metrics = {
        "dists": dists.tolist(),
        "mass_orig": rdf_raw.tolist(),
        "mass_dec": rdf_dec.tolist()
    }
    output_metrics(metrics)

    sys.exit(0)

# ------------ Top-level: 运行 pressio -i 3 次 (x,y,z)，再 RDF → dists ------------
# 用法: python run_pressio_rdf_pipeline.py --external_mode -i <prefix> --dim 7852 --dim 1037
# prefix 对应 <prefix>.x.f32.dat, <prefix>.y.f32.dat, <prefix>.z.f32.dat
if args.external_mode and input_file and len(dims) >= 2:
    dec_dir = os.path.join(script_dir, "decompressed")
    os.makedirs(dec_dir, exist_ok=True)
    nt, na = dims[0], dims[1]
    input_prefix = original_input or input_file
    input_prefix = input_prefix.replace(".x.f32.dat", "").replace(".y.f32.dat", "").replace(".z.f32.dat", "").rstrip(".")
    for axis in "xyz":
        inp = f"{input_prefix}.{axis}.f32.dat"
        out = os.path.join(dec_dir, f"{axis}.bin")
        if not os.path.isfile(inp):
            print("Usage: --external_mode -i <prefix> --dim <nt> --dim <na>  (prefix.x/y/z.f32.dat)", file=sys.stderr)
            output_metrics({"dists": []})
            sys.exit(1)
        pressio_cmd = [
            pressio,
            "-i", inp,
            "-d", str(nt), "-d", str(na),
            "-t", "float",
            "-b", f"compressor={compressor}",
            "-o", f"rel={rel}",
            "-W", out,
        ]
        result = subprocess.run(pressio_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            output_metrics({"dists": []})
            sys.exit(1)
    dec_x = os.path.join(dec_dir, "x.bin")
    dec_y = os.path.join(dec_dir, "y.bin")
    dec_z = os.path.join(dec_dir, "z.bin")
    rdf_cmd = [
        sys.executable,
        rdf_script,
        "--standalone",
        "--raw_prefix", input_prefix,
        "--nt", str(nt),
        "--na", str(na),
        "--decompressed_x", dec_x,
        "--decompressed_y", dec_y,
        "--decompressed_z", dec_z,
    ]
    result = subprocess.run(rdf_cmd, capture_output=True, text=True)
    if result.stderr:
        sys.stderr.write(result.stderr)
    if result.returncode != 0:
        output_metrics({"dists": []})
        sys.exit(result.returncode)
    dists_path = os.path.join(os.path.dirname(rdf_script), "dists.npy")
    rdf_raw_path = os.path.join(os.path.dirname(rdf_script), "rdf_raw.npy")
    rdf_dec_path = os.path.join(os.path.dirname(rdf_script), "rdf_dec.npy")
    if not os.path.isfile(dists_path):
        output_metrics({"dists": []})
        sys.exit(1)
    dists = np.load(dists_path)
    rdf_raw = np.load(rdf_raw_path)
    rdf_dec = np.load(rdf_dec_path)
    metrics = {
        "dists": dists.tolist(),
        "mass_orig": rdf_raw.tolist(),
        "mass_dec": rdf_dec.tolist()
    }
    output_metrics(metrics)
    sys.exit(0)

# 未匹配任何模式
output_metrics({"dists": []})
sys.exit(1)
