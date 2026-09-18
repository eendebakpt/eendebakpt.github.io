"""Benchmark dispatched methods that benefit from the fast dispatcher.

Runs a fixed set of micro-benchmarks with ``timeit`` and writes the results
as JSON.  The script only depends on numpy and the standard library, so it
can be executed against any numpy version, e.g. via uv:

    uv run --python 3.13 --no-project --with numpy==2.5.1 bench_dispatched.py \
        --output results/numpy-2.5.1-dispatched.json

The cases are chosen to exercise dispatched methods that gain performance from
the fast dispatcher mechanism (PR #32523), which bypasses the slow
``__array_function__`` machinery for pure ndarray operations.
"""

import argparse
import json
import platform
import sys
import timeit

SETUP = """
import numpy as np

# Small arrays
a = np.arange(10.0)
f = np.arange(10.0)
m = np.arange(9.0).reshape(3, 3)

# Masks and indices
mask = np.array([True, False, True, False, True, False, True, False, True, False])
indices = np.array([1, 2, 5, 8])

# Search array
sf = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
"""

# (group, name, statement)
CASES = [
    # --- shape/structure methods ---
    ("shape", "np.flatten(m)", "np.flatten(m)"),
    ("shape", "np.ravel(m)", "np.ravel(m)"),
    ("shape", "np.transpose(m)", "np.transpose(m)"),
    ("shape", "np.reshape(a, (2, 5))", "np.reshape(a, (2, 5))"),
    ("shape", "np.swapaxes(m, 0, 1)", "np.swapaxes(m, 0, 1)"),
    
    # --- reduction-like dispatched methods ---
    ("reduction_dispatch", "np.cumsum(a)", "np.cumsum(a)"),
    ("reduction_dispatch", "np.cumprod(a)", "np.cumprod(a)"),
    ("reduction_dispatch", "np.argmin(f)", "np.argmin(f)"),
    ("reduction_dispatch", "np.argmax(f)", "np.argmax(f)"),
    ("reduction_dispatch", "np.nonzero(a)", "np.nonzero(a)"),
    ("reduction_dispatch", "np.flatnonzero(a)", "np.flatnonzero(a)"),
    
    # --- selection/manipulation methods ---
    ("selection", "np.take(a, [1, 2])", "np.take(a, [1, 2])"),
    ("selection", "np.extract(mask, f)", "np.extract(mask, f)"),
    ("selection", "np.compress([T,F]*5, a)", "np.compress([True, False]*5, a)"),
    ("selection", "np.argsort(f)", "np.argsort(f)"),
    ("selection", "np.searchsorted(sf, 5.5)", "np.searchsorted(sf, 5.5)"),
    
    # --- other dispatched methods ---
    ("other", "np.round(f)", "np.round(f)"),
    ("other", "np.repeat(a, 2)", "np.repeat(a, 2)"),
]

REPEAT = 9
TARGET_TIME = 0.02  # seconds per repeat


def run_case(stmt: str) -> float:
    """Return the best time per call in nanoseconds."""
    timer = timeit.Timer(stmt, setup=SETUP)
    number, elapsed = timer.autorange()
    # rescale so a single repeat takes roughly TARGET_TIME seconds
    number = max(10, int(number * TARGET_TIME / elapsed))
    times = timer.repeat(repeat=REPEAT, number=number)
    return min(times) / number * 1e9


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", default=None,
                        help="write JSON results to this file (default: stdout)")
    parser.add_argument("--label", default=None,
                        help="label for this run (default: numpy version)")
    args = parser.parse_args()

    import numpy as np

    results = []
    for group, name, stmt in CASES:
        ns = run_case(stmt)
        results.append({"group": group, "name": name, "ns": round(ns, 1)})
        print(f"  {group:>18} | {name:<24} {ns:9.1f} ns", file=sys.stderr)

    data = {
        "label": args.label or np.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
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
