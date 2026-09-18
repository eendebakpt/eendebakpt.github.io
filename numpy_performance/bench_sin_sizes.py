"""Benchmark np.sin with varying array sizes to measure per-call vs per-element overhead.

Runs np.sin(x) on float64 arrays of sizes 1, 2, 4, ..., 2^27 and records
the *total* time and *per-element* time. The binary sizes help reveal when
per-call overhead dominates vs. when per-element computation takes over.

The input is ``np.linspace(0, 2*pi, size)``: bounded values in sorted order,
independent of the array size. Two alternatives distort the per-element cost:

* ``np.arange(size)`` makes the argument magnitude grow with the size, and
  sin() range reduction is ~5x slower for arguments above ~1e8, so the
  largest arrays come out slower per element.
* Random values (e.g. ``rng.uniform(0, 2*pi, size)``) are ~2.4x slower per
  element than the same values sorted, because a data-dependent branch in the
  SIMD sin loop mispredicts. The penalty only shows up above ~4096 elements:
  for smaller arrays the timing loop repeats the same data and the branch
  predictor memorizes the pattern.

Unless ``--no-malloc-tuning`` is given, memory allocation is made
reproducible: glibc malloc is configured (via ``mallopt``) to never use mmap
and never trim the heap, and NumPy's transparent-hugepage advice is disabled
(``NUMPY_MADVISE_HUGEPAGE=0``). Each np.sin call allocates a fresh output
array; with default settings that memory is returned to the OS after every
call and page-faulted in again on the next one. The page-fault cost depends on
the size and on transparent-hugepage state (on the test machine it was worst,
about +5 ns/elem, for heap-backed 4-16 MB arrays with hugepage advice), which
distorts the per-element cost. With the tuning, repeated calls reuse
already-faulted memory and the measured time is per-call overhead plus
computation.

The chart in the blog post is pinned to one NumPy build (main at e765e67,
2026-09-18); plot_sin_overhead.py reads exactly this results file:

    taskset -c 2 uv run --python 3.13 --no-project \
        --with "numpy @ git+https://github.com/numpy/numpy@e765e6725e25dbf9ab28c84914e4dd9edc9e4548" \
        bench_sin_sizes.py --label main-e765e67 \
        --output results/numpy-2.6dev-20260918-e765e67-sin-sizes.json
"""

import argparse
import ctypes
import json
import os
import platform
import sys
import timeit

REPEAT = 5
TARGET_TIME = 0.05  # seconds per repeat


def tune_malloc() -> bool:
    """Make glibc malloc keep freed memory instead of returning it to the OS."""
    try:
        libc = ctypes.CDLL(None)
        mallopt = libc.mallopt
    except (OSError, AttributeError):
        return False
    M_TRIM_THRESHOLD, M_TOP_PAD, M_MMAP_MAX = -1, -2, -4
    ok = mallopt(M_MMAP_MAX, 0)  # never use mmap for large allocations
    ok &= mallopt(M_TRIM_THRESHOLD, 2**31 - 1)  # never give heap memory back
    ok &= mallopt(M_TOP_PAD, 64 * 2**20)
    return bool(ok)


def run_sin_size(size: int) -> tuple[float, float]:
    """
    Return (total_ns, per_element_ns) for np.sin on an array of given size.

    Each of the REPEAT rounds re-creates the input array in a fresh setup and
    runs the operation enough times to reach TARGET_TIME; the minimum over
    the rounds is reported. Re-creating the array matters for the largest
    sizes: on the test machine, buffers of 512 MB and up were sporadically
    ~50% slower depending on where they landed in memory (no page faults
    involved), and the minimum over placements is the representative cost.
    """
    setup = f"""
import numpy as np
a = np.linspace(0.0, 2 * np.pi, {size})
"""
    stmt = "np.sin(a)"

    timer = timeit.Timer(stmt, setup=setup)
    number, elapsed = timer.autorange()
    # rescale so a single round takes roughly TARGET_TIME seconds
    number = max(1, int(number * TARGET_TIME / elapsed))
    times = [timeit.Timer(stmt, setup=setup).timeit(number=number)
             for _ in range(REPEAT)]

    total_ns = min(times) / number * 1e9
    per_element_ns = total_ns / size if size > 0 else 0
    return total_ns, per_element_ns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", default=None,
                        help="write JSON results to this file (default: stdout)")
    parser.add_argument("--label", default=None,
                        help="label for this run (default: numpy version)")
    parser.add_argument("--min-power", type=int, default=0,
                        help="smallest array size is 2**MIN_POWER (default: 0)")
    parser.add_argument("--max-power", type=int, default=27,
                        help="largest array size is 2**MAX_POWER (default: 27)")
    parser.add_argument("--no-malloc-tuning", action="store_true",
                        help="leave glibc malloc at its defaults (see module docstring)")
    args = parser.parse_args()

    malloc_tuned = False
    if not args.no_malloc_tuning:
        # must be set before numpy is first imported
        os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"
        malloc_tuned = tune_malloc()
        if not malloc_tuned:
            print("warning: could not tune malloc (not glibc?)", file=sys.stderr)

    import numpy as np

    # Array sizes: 2^min_power, ..., 2^max_power (default 1 ... 2^27 > 100M elements)
    sizes = [2 ** i for i in range(args.min_power, args.max_power + 1)]
    
    results = []
    for size in sizes:
        total_ns, per_elem_ns = run_sin_size(size)
        results.append({
            "size": size,
            "total_ns": round(total_ns, 1),
            "per_element_ns": round(per_elem_ns, 2),
        })
        print(f"  np.sin(array[{size:8d}]): {total_ns:12.1f} ns total, "
              f"{per_elem_ns:8.2f} ns/elem", file=sys.stderr)

    data = {
        "label": args.label or np.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "operation": "np.sin(float64_array)",
        "input": "linspace(0, 2pi, size)",
        "malloc_tuned": malloc_tuned,
        "results": results,
    }
    text = json.dumps(data, indent=2)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
