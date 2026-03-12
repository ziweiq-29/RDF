#!/usr/bin/env python3
"""
RDF/EXAALT：对 x/y/z 各跑一次 pressio（qoi:metric=external），
external command 为 run_pressio_rdf_pipeline.py --external_mode --axis {x|y|z}。
pressio 压缩/解压后传 --input/--decompressed，pipeline 写 rdf_metrics.json，
stdout 只打 metrics_file，由 qoi.cc 读 vector 算 QOI。
"""
import os
os.environ.pop("PYTHONPATH", None)
os.environ["PYTHONNOUSERSITE"] = "1"
import argparse
import subprocess
import sys

RDF_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.path.join(RDF_DIR, "run_pressio_rdf_pipeline.py")


def main():
    p = argparse.ArgumentParser(
        description="Run pressio 3× (x,y,z) with qoi:metric=external; pipeline outputs metrics_file for qoi.cc.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--compressor", default="sz3", help="Compressor name for pressio (default: sz3)")
    p.add_argument("--prefix", required=True, help="Path prefix for .x/.y/.z.f32.dat")
    p.add_argument("--nt", type=int, required=True, help="First dimension (e.g. 7852)")
    p.add_argument("--na", type=int, required=True, help="Second dimension (e.g. 1037)")
    p.add_argument("--rel", default="1e-4", help="Compressor rel (e.g. 1e-4)")
    p.add_argument("--pressio", default=os.environ.get("PRESSIO_CMD", "pressio"),
                   help="Pressio executable (default: pressio or PRESSIO_CMD)")
    p.add_argument("--pressio-opts", dest="pressio_opts", action="append", default=[],
                   help="Extra pressio options as key=value. Can repeat.")
    p.add_argument("--print-cmd", action="store_true", help="Print the pressio command for first axis and exit.")
    p.add_argument("--clean-tmp", action="store_true",
                   help="Remove RDF/rdf_tmp/*.bin before run to avoid stale truncated bins from previous jobs.")
    args = p.parse_args()

    prefix = args.prefix.rstrip(".")
    if args.clean_tmp:
        import glob
        tmp_dir = os.path.join(RDF_DIR, "rdf_tmp")
        for f in glob.glob(os.path.join(tmp_dir, "*.bin")):
            try:
                os.remove(f)
            except OSError:
                pass
    python_exe = sys.executable
    pipeline_opts = f"--original_input {prefix} --dim {args.nt} --dim {args.na} --compressor {args.compressor} --rel {args.rel} --pressio {args.pressio}"
    for opt in args.pressio_opts:
        pipeline_opts += f" --pressio-opts {opt}"

    base_opts = [
        "-d", str(args.nt), "-d", str(args.na),
        "-t", "float",
        "-b", f"compressor={args.compressor}",
        "-o", f"rel={args.rel}",
    ]
    for opt in args.pressio_opts:
        base_opts.extend(["-o", opt])
    external_opts = [
        "-b", "qoi:metric=external",
        "-b", "external:launch_metric=print",
        "-o", "external:use_many=1",
        "-m", "qoi", "-M", "all",
    ]

    for axis in "xyz":
        ext_cmd = f"{python_exe} {PIPELINE} --external_mode --axis {axis} {pipeline_opts}"
        cmd = [
            args.pressio,
            "-i", f"{prefix}.{axis}.f32.dat",
        ] + base_opts + external_opts + ["-o", f"external:command={ext_cmd}"]
        if args.print_cmd:
            print(" ".join(cmd))
            sys.exit(0)
        if axis == "z":
            r = subprocess.run(cmd, cwd=RDF_DIR)
        else:
            r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=sys.stderr, cwd=RDF_DIR)
        if r.returncode != 0:
            sys.exit(r.returncode)
    sys.exit(0)


if __name__ == "__main__":
    main()
