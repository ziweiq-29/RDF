#!/bin/bash
# RDF 运行环境：module + PYTHONPATH + 依赖。每次登录新节点执行一次：
#   source /anvil/projects/x-cis240669/RDF/env.sh
# 必须用 source，不能 ./env.sh
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Usage: source $(basename "$0")   # do not run as ./env.sh" >&2
  exit 1
fi

RDF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# ------------ Module 环境 ------------
module purge 2>/dev/null || true
module load gcc/11.2.0 python 2>/dev/null || true
# 若集群有 pressio 模块，取消下面某一行的注释（或按实际模块名修改）
# module load pressio 2>/dev/null || true
# module load libpressio 2>/dev/null || true

# ------------ Python 路径（优先用 pip 安装的包，避免与系统 numpy/scipy 冲突） ------------
export PYTHONPATH="$("${PYTHON:-python}" -c 'import site; print(site.getusersitepackages())')${PYTHONPATH:+:$PYTHONPATH}"

# ------------ 依赖：numpy / scipy / scikit-learn 用 pip 装到 --user，版本一致避免 ABI 错误 ------------
"${PYTHON:-python}" -m pip install --user 'numpy>=1.20' 'scipy' 'scikit-learn' -q 2>/dev/null || true

# ------------ 可选：pressio 不在 PATH 时，指定可执行文件路径 ------------
# export PRESSIO_CMD=/path/to/pressio
