
#!/usr/bin/env python3
"""
跑 3 次 pressio (x,y,z)，得到 qoi 需要的 dists。
用法: python run_pressio_external.py --prefix <path> --nt <n> --na <n> [--rel 1e-4] [--qoi-out ...]
运行前可 source env.sh 或设置 PRESSIO_CMD。
"""
import argparse
import os
import subprocess
import sys

RDF_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.path.join(RDF_DIR, "run_pressio_rdf_pipeline.py")
DEFAULT_QOI_OUT = os.path.join(RDF_DIR, "dists_qoi_output.txt")


def main():
    p = argparse.ArgumentParser(
        description="Run pressio 3× (x,y,z), output dists for qoi.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--compressor", default="sz3", help="Compressor name for pressio (default: sz3)")
    p.add_argument("--prefix", required=True, help="Path prefix for .x/.y/.z.f32.dat")
    p.add_argument("--nt", type=int, required=True, help="First dimension (e.g. 7852)")
    p.add_argument("--na", type=int, required=True, help="Second dimension (e.g. 1037)")
    p.add_argument("--rel", default="1e-4", help="Compressor rel (e.g. 1e-4)")
    p.add_argument("--pressio-cmd", dest="pressio_cmd", default=os.environ.get("PRESSIO_CMD", "pressio"),
                   help="Pressio executable (default: pressio or PRESSIO_CMD)")
    p.add_argument("--pressio-opts", dest="pressio_opts", action="append", default=[],
                   help="Extra pressio/sz3 options as key=value (e.g. sz3:algorithm_str=ALGO_BIOMD). Can repeat.")
    p.add_argument("--print-cmd", action="store_true", help="Print the pressio command for the first axis and exit (for debugging).")
    args = p.parse_args()

    pressio = args.pressio_cmd
    compressor = args.compressor
    prefix = args.prefix.rstrip(".")
    pressio_opts = list(args.pressio_opts)
    base_opts = [
        "-d", str(args.nt), "-d", str(args.na),
        "-t", "float",
        "-b", f"compressor={compressor}", "-o", f"rel={args.rel}",
    ]
    for opt in pressio_opts:
        base_opts.extend(["-o", opt])
    pipeline_opts = f"--original_input {prefix} --compressor {compressor} --rel {args.rel}"
    for opt in pressio_opts:
        pipeline_opts += f" --pressio-opts {opt}"
    base_opts.extend([
        "-b", "qoi:metric=external",
        "-o", f"external:command=python {PIPELINE} --external_mode {pipeline_opts}",
        "-b", "external:launch_metric=print", "-o", "external:use_many=1",
        "-m", "qoi", "-M", "all",
    ])

    for axis in "xyz":
        cmd = [pressio, "-i", f"{prefix}.{axis}.f32.dat"] + base_opts
        if args.print_cmd:
            print(" ".join(cmd))
            sys.exit(0)
        if axis == "z":
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=RDF_DIR)
            out = r.stdout or ""
            err = r.stderr or ""
            if err:
                sys.stderr.write(err)
            sys.stdout.write(out)
            sys.stdout.flush()
        else:
            r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=sys.stderr, cwd=RDF_DIR)
        if r.returncode != 0:
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
