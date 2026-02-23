"""
RDF on EXAALT: raw vs decompressed → dists.
用法：--standalone --decompressed_x/y/z <path>，读三份解压文件算 RDF，输出 dists。
Uses freud when RDF_USE_FREUD=1; else NumPy RDF.
"""
import argparse
import json
import os
import sys
import numpy as np
from numba import njit


_USE_FREUD = False
try:
    if os.environ.get("RDF_USE_FREUD", "").lower() in ("1", "true", "yes"):
        from freud.box import Box
        from freud.density import RDF
        _USE_FREUD = True
except Exception:
    pass

timestep = 0
DEFAULT_NT = 7852
DEFAULT_NA = 1037
DEFAULT_RAW_PREFIX = "/anvil/projects/x-cis240669/EXAALT/SDRBENCH-exaalt-helium/dataset1-7852x1037"


def read_raw_frame(prefix, t, nt, na):
    def load_axis(suffix):
        arr = np.fromfile(f"{prefix}.{suffix}.f32.dat", dtype=np.float32)
        return arr.reshape(nt, na)
    x = load_axis("x")[t]
    y = load_axis("y")[t]
    z = load_axis("z")[t]
    return np.stack([x, y, z], axis=1).astype(np.float32)


# def _compute_rdf_numpy(pos, bins=300):
#     """Pure NumPy RDF (same interface as freud path)."""
#     pos = np.asarray(pos, dtype=np.float64)
#     mins = pos.min(axis=0)
#     pos = pos - mins
#     L = pos.max(axis=0)
#     rmax = min(L) / 2
#     N = pos.shape[0]
#     rho = N / (L[0] * L[1] * L[2])
#     edges = np.linspace(0, rmax, bins + 1)
#     dr = edges[1] - edges[0]
#     # Pairwise distances (upper triangle, no self)
#     d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2))
#     d = d[np.triu_indices(N, k=1)]
#     hist, _ = np.histogram(d, bins=edges)
#     hist = hist.astype(np.float64)
#     r = (edges[:-1] + edges[1:]) / 2
#     vol_shell = 4 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3) / 3
#     ideal = (N / 2) * rho * vol_shell
#     np.maximum(ideal, 1e-300, out=ideal)
#     g = hist / ideal
#     return g


from numba import njit
import numpy as np

@njit
def _accumulate_hist(pos, ref_indices, hist, L, rmax, dr):
    """
    Cell-list based RDF accumulation.
    Drop-in replacement for brute-force version.
    """

    N = pos.shape[0]
    bins = hist.shape[0]

    # -----------------------------
    # Build cell list
    # -----------------------------
    cell_size = rmax / 10   # ⭐ 可以调大/小，5比较稳
    nx = int(L[0] / cell_size) + 1
    ny = int(L[1] / cell_size) + 1
    nz = int(L[2] / cell_size) + 1

    ncells = nx * ny * nz

    head = -np.ones(ncells, dtype=np.int64)
    nextp = np.empty(N, dtype=np.int64)

    # assign particles to cells
    for i in range(N):
        cx = int(pos[i,0] / cell_size)
        cy = int(pos[i,1] / cell_size)
        cz = int(pos[i,2] / cell_size)

        cid = (cz * ny + cy) * nx + cx

        nextp[i] = head[cid]
        head[cid] = i

    # -----------------------------
    # accumulate RDF
    # -----------------------------
    invL0 = 1.0 / L[0]
    invL1 = 1.0 / L[1]
    invL2 = 1.0 / L[2]

    for ii in range(len(ref_indices)):
        i = ref_indices[ii]

        cx = int(pos[i,0] / cell_size)
        cy = int(pos[i,1] / cell_size)
        cz = int(pos[i,2] / cell_size)

        # search neighbor cells
        for dx in (-1,0,1):
            x2 = cx + dx
            if x2 < 0: x2 += nx
            elif x2 >= nx: x2 -= nx

            for dy in (-1,0,1):
                y2 = cy + dy
                if y2 < 0: y2 += ny
                elif y2 >= ny: y2 -= ny

                for dz in (-1,0,1):
                    z2 = cz + dz
                    if z2 < 0: z2 += nz
                    elif z2 >= nz: z2 -= nz

                    cid = (z2 * ny + y2) * nx + x2

                    j = head[cid]
                    while j != -1:

                        if j != i:

                            dx = pos[i,0] - pos[j,0]
                            dy = pos[i,1] - pos[j,1]
                            dz = pos[i,2] - pos[j,2]

                            # minimum image
                            dx -= L[0] * np.rint(dx * invL0)
                            dy -= L[1] * np.rint(dy * invL1)
                            dz -= L[2] * np.rint(dz * invL2)

                            r2 = dx*dx + dy*dy + dz*dz
                            if r2 < rmax*rmax:
                                r = np.sqrt(r2)
                                b = int(r / dr)
                                if b < bins:
                                    hist[b] += 1.0

                        j = nextp[j]

def compute_rdf_numba(pos, bins=300, sample_ratio=0.01, seed=0):
    """
    Fast RDF for large MD datasets (EXAALT-scale).
    """
    pos = np.asarray(pos, dtype=np.float64)

    mins = pos.min(axis=0)
    pos -= mins

    L = pos.max(axis=0)
    N = pos.shape[0]

    rmax = min(L) / 2
    rho = N / (L[0] * L[1] * L[2])

    edges = np.linspace(0, rmax, bins + 1)
    dr = edges[1] - edges[0]

    hist = np.zeros(bins, dtype=np.float64)

    # -----------------------------
    # reference particle sampling
    # -----------------------------
    rng = np.random.default_rng(seed)

    n_ref = max(1, int(N * sample_ratio))
    ref_indices = rng.choice(N, n_ref, replace=False).astype(np.int64)

    # -----------------------------
    # numba histogram accumulation
    # -----------------------------
    _accumulate_hist(pos, ref_indices, hist, L, rmax, dr)

    # -----------------------------
    # normalization
    # -----------------------------
    r = (edges[:-1] + edges[1:]) / 2
    vol_shell = 4 * np.pi * (edges[1:]**3 - edges[:-1]**3) / 3

    ideal = n_ref * rho * vol_shell
    np.maximum(ideal, 1e-300, out=ideal)

    g = hist / ideal

    return g
def _compute_rdf_numpy(pos, bins=300, sample_ratio=0.01):
    pos = np.asarray(pos, dtype=np.float64)

    mins = pos.min(axis=0)
    pos = pos - mins

    L = pos.max(axis=0)
    rmax = min(L) / 2
    N = pos.shape[0]
    rho = N / (L[0] * L[1] * L[2])

    edges = np.linspace(0, rmax, bins + 1)
    dr = edges[1] - edges[0]

    hist = np.zeros(bins, dtype=np.float64)

    # -----------------------------
    # ⭐ Reference particle sampling
    # -----------------------------
    n_ref = int(N * sample_ratio)
    ref_indices = np.random.choice(N, n_ref, replace=False)
    ref_mask = np.zeros(N, dtype=bool)
    ref_mask[ref_indices] = True

    # -----------------------------
    # Build cell list
    # -----------------------------
    cell_size = rmax / 10
    ncell = np.maximum((L / cell_size).astype(int), 1)

    cells = {}
    for i, p in enumerate(pos):
        idx = tuple((p / cell_size).astype(int))
        cells.setdefault(idx, []).append(i)

    offsets = [(dx, dy, dz)
               for dx in (-1, 0, 1)
               for dy in (-1, 0, 1)
               for dz in (-1, 0, 1)]

    # -----------------------------
    # Accumulate histogram
    # -----------------------------
    for cell_idx, plist in cells.items():
        for dx, dy, dz in offsets:
            neigh_idx = (
                (cell_idx[0] + dx) % ncell[0],
                (cell_idx[1] + dy) % ncell[1],
                (cell_idx[2] + dz) % ncell[2],
            )
            if neigh_idx not in cells:
                continue

            qlist = cells[neigh_idx]

            for i in plist:
                if not ref_mask[i]:   # ⭐ only reference particles
                    continue

                for j in qlist:
                    if j <= i:
                        continue

                    rij = pos[i] - pos[j]
                    rij -= L * np.round(rij / L)

                    r = np.linalg.norm(rij)
                    if r < rmax:
                        bin_idx = int(r / dr)
                        if bin_idx < bins:
                            hist[bin_idx] += 1  # ⭐ 不再乘2

    # -----------------------------
    # Normalization
    # -----------------------------
    r = (edges[:-1] + edges[1:]) / 2
    vol_shell = 4 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3) / 3

    # ⭐ normalization must use N_ref
    ideal = n_ref * rho * vol_shell

    np.maximum(ideal, 1e-300, out=ideal)
    g = hist / ideal

    return g


def compute_rdf(pos, bins=300,sample_ratio=None):
    
    pos = np.asarray(pos, dtype=np.float32)
    mins = pos.min(axis=0)
    pos = pos - mins
    L = pos.max(axis=0)
    box_length = max(L)
    rmax = min(L) / 2
    N = pos.shape[0]

    if sample_ratio is None:
        if N < 20000:
            sample_ratio = 1.0        # small systems → no sampling
        elif N < 200000:
            sample_ratio = 0.1        # medium systems
        else:
            sample_ratio = 0.01 
    if _USE_FREUD:
        box = Box.cube(box_length)
        rdf = RDF(bins=bins, r_max=rmax)
        rdf.compute((box, pos))
        return np.asarray(rdf.rdf, dtype=np.float64)
    # return _compute_rdf_numpy(pos.astype(np.float64), bins=bins)
    return compute_rdf_numba(
        pos.astype(np.float64),
        bins=bins,
        sample_ratio=sample_ratio
    )


def rdf_distance(g1, g2):
    dists=np.abs(g1-g2)
    return dists

def main():
    parser = argparse.ArgumentParser(description="RDF: raw vs decompressed → dists")
    parser.add_argument("--standalone", action="store_true", required=True, help="必须：读三份解压文件")
    parser.add_argument("--decompressed_x", required=True, help="解压后的 x 文件")
    parser.add_argument("--decompressed_y", required=True, help="解压后的 y 文件")
    parser.add_argument("--decompressed_z", required=True, help="解压后的 z 文件")
    parser.add_argument("--raw_prefix", default=DEFAULT_RAW_PREFIX, help=f"Raw .x/.y/.z.f32.dat 前缀 (default: {DEFAULT_RAW_PREFIX})")
    parser.add_argument("--nt", type=int, default=DEFAULT_NT, help=f"时间步数 (default: {DEFAULT_NT})")
    parser.add_argument("--na", type=int, default=DEFAULT_NA, help=f"原子数 (default: {DEFAULT_NA})")
    args = parser.parse_args()

    nt, na = args.nt, args.na
    raw_prefix = args.raw_prefix

    for k in ("decompressed_x", "decompressed_y", "decompressed_z"):
        path = getattr(args, k)
        if not path or not os.path.isfile(path):
            print(f"Missing or not a file: --{k}", file=sys.stderr)
            sys.exit(1)

    coords_dec = np.stack([
        np.fromfile(args.decompressed_x, dtype=np.float32).reshape(nt, na)[timestep],
        np.fromfile(args.decompressed_y, dtype=np.float32).reshape(nt, na)[timestep],
        np.fromfile(args.decompressed_z, dtype=np.float32).reshape(nt, na)[timestep],
    ], axis=1).astype(np.float32)
    coords_raw = read_raw_frame(raw_prefix, timestep, nt, na)
    rdf_raw = compute_rdf(coords_raw)
    rdf_dec = compute_rdf(coords_dec)
    dists = rdf_distance(rdf_raw , rdf_dec)
    # rdf_dir = os.path.dirname(os.path.abspath(__file__))
    np.save(os.path.join("dists.npy"), dists)
    np.save(os.path.join("rdf_raw.npy"), rdf_raw)
    np.save(os.path.join("rdf_dec.npy"), rdf_dec)


if __name__ == "__main__":
    main()