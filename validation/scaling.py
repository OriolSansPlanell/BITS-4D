"""Time and peak memory of the spatial refinement against volume size.

    python -m validation.scaling                # 64³ .. 256³, 5 classes
    python -m validation.scaling --sizes 64 128 --classes 9

Peak memory is measured with ``tracemalloc`` (numpy reports its buffers to
it), so it counts the arrays the solver allocates, not the interpreter.
Numbers depend on the machine; the ratios between solvers do not.
"""

from __future__ import annotations

import argparse
import time
import tracemalloc

import numpy as np

from model.spatial_prior import ROIDerivedMRF, UnaryScores, potts_cost


def _problem(size, n_classes, seed=0):
    rng = np.random.default_rng(seed)
    shape = (size, size, size)
    truth = (np.arange(size)[None, None, :] * n_classes // size).astype(np.int32)
    truth = np.broadcast_to(truth, shape)
    bins = 256
    table = rng.normal(size=(bins, n_classes)).astype(np.float32)
    table[np.arange(bins), np.arange(bins) * n_classes // bins] += 1.5
    per_class = bins // n_classes
    rows = (truth * per_class + rng.integers(0, per_class, shape)).astype(np.int32)
    noise = rng.random(shape) < 0.3
    rows[noise] = rng.integers(0, bins, int(noise.sum()))
    neutron = truth * 100.0 + rng.normal(0, 40, shape).astype(np.float32)
    xray = truth * 60.0 + rng.normal(0, 40, shape).astype(np.float32)
    return UnaryScores(table, rows), neutron.astype(np.float32), xray.astype(np.float32)


def measure(size, n_classes, method, slab_thickness=None, n_sweeps=5):
    scores, neutron, xray = _problem(size, n_classes)
    mrf = ROIDerivedMRF(beta=1.0, n_sweeps=n_sweeps, memory_budget_gb=1e6)
    mrf.pairwise = potts_cost(n_classes)
    tracemalloc.start()
    start = time.perf_counter()
    labels, info = mrf.refine(scores, neutron, xray, method=method,
                              slab_thickness=slab_thickness)
    seconds = time.perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    voxels = size ** 3
    return {"size": f"{size}³", "voxels": voxels, "classes": n_classes,
            "solver": method + (f", slabs of {slab_thickness}"
                                if slab_thickness else ""),
            "seconds": round(seconds, 2),
            "peak_MiB": round(peak / 2 ** 20, 1),
            "bytes_per_voxel": round(peak / voxels, 1),
            "ns_per_voxel_sweep": round(1e9 * seconds / voxels / n_sweeps, 1)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", nargs="*", type=int, default=[64, 128, 192, 256])
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--slab", type=int, default=32)
    args = parser.parse_args(argv)
    rows = []
    for size in args.sizes:
        for method, slab in (("mean_field", None), ("mean_field", args.slab),
                             ("icm", None)):
            if slab and slab >= size:
                continue
            rows.append(measure(size, args.classes, method, slab))
            print(rows[-1], flush=True)
    header = list(rows[0])
    print("\n| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(str(row[key]) for key in header) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
