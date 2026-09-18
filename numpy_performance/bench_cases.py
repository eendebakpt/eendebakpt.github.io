"""Benchmark scalar and small-array performance of the installed numpy.

Runs a fixed set of micro-benchmarks with ``timeit`` and writes the results
as JSON.  The script only depends on numpy and the standard library, so it
can be executed against any numpy version, e.g. via uv:

    uv run --python 3.13 --no-project --with numpy==2.4.6 bench_cases.py \
        --output results/numpy-2.4.6.json

The cases are chosen to exercise the code paths touched by the
scalar/small-array performance PRs listed in the accompanying blog post:

* scalar ufunc calls and scalar math   (gh-21527, gh-29819, gh-27901)
* small-array ufunc calls              (gh-23020, gh-30593, gh-31068)
* reductions on small and medium arrays (gh-28123, gh-28148, gh-31274, gh-31845)
* call/creation overhead               (gh-28710, gh-29656)

Operands: ``a``, ``b`` are float64 arrays of shape (10,), ``a1000`` has shape
(1000,), ``f64``/``i64`` are NumPy scalars and ``m3`` is a 3x3 float64 matrix.
"""

import argparse
import json
import platform
import sys
import timeit

SETUP = """
import copy
import numpy as np

a = np.arange(10.0)
b = np.ones(10)
out = np.empty(10)
bool10 = a > 4
i10 = np.arange(10)
a1000 = np.arange(1000.0)
bool1000 = a1000 < 2000.0
f = np.float64(1.5)
g = np.float64(2.5)
i = np.int64(3)
j = np.int64(7)
bi = np.ones(10, dtype=np.int64)
m3 = np.array([[2.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 4.0]])
"""

# (group, name, statement)
CASES = [
    # --- scalar ufunc calls and scalar math ---
    ("scalar", "np.add(1.5, 2.5)", "np.add(1.5, 2.5)"),
    ("scalar", "np.multiply(f64, f64)", "np.multiply(f, g)"),
    ("scalar", "np.sqrt(f64)", "np.sqrt(f)"),
    ("scalar", "np.sin(f64)", "np.sin(f)"),
    ("scalar", "f64 + f64", "f + g"),
    ("scalar", "f64 ** 2", "f ** 2"),
    ("scalar", "np.float64(1.5)", "np.float64(1.5)"),
    ("scalar", "copy.copy(f64)", "copy.copy(f)"),
    ("scalar", "i64 + i64", "i + j"),
    ("scalar", "np.abs(i64)", "np.abs(i)"),
    ("scalar", "np.multiply(i64, i64)", "np.multiply(i, j)"),
    # --- small-array ufunc calls (n=10) ---
    ("ufunc-small", "a + b", "a + b"),
    ("ufunc-small", "a * 2.0", "a * 2.0"),
    ("ufunc-small", "np.add(a, b)", "np.add(a, b)"),
    ("ufunc-small", "np.add(a, b, out=out)", "np.add(a, b, out=out)"),
    ("ufunc-small", "np.sqrt(a)", "np.sqrt(a)"),
    ("ufunc-small", "np.exp(a)", "np.exp(a)"),
    ("ufunc-small", "a > b", "a > b"),
    ("ufunc-small", "np.maximum(a, b)", "np.maximum(a, b)"),
    ("ufunc-small", "ai + bi (int64)", "i10 + bi"),
    # --- reductions on a small array (n=10) ---
    ("reduction-10", "np.sum", "np.sum(a)"),
    ("reduction-10", "np.mean", "np.mean(a)"),
    ("reduction-10", "np.any", "np.any(bool10)"),
    ("reduction-10", "np.max", "np.max(a)"),
    ("reduction-10", "np.sum (int64)", "np.sum(i10)"),
    # --- reductions on a contiguous medium array (n=1000) ---
    ("reduction-1000", "np.sum", "np.sum(a1000)"),
    ("reduction-1000", "np.mean", "np.mean(a1000)"),
    ("reduction-1000", "np.any", "np.any(bool1000)"),
    ("reduction-1000", "np.max", "np.max(a1000)"),
    # --- np.nonzero / np.count_nonzero ---
    ("nonzero", "np.nonzero(a) (n=10)", "np.nonzero(a)"),
    ("nonzero", "np.nonzero(bool10)", "np.nonzero(bool10)"),
    ("nonzero", "np.nonzero(bool1000)", "np.nonzero(bool1000)"),
    ("nonzero", "np.count_nonzero(a) (n=10)", "np.count_nonzero(a)"),
    ("nonzero", "np.count_nonzero(bool1000)", "np.count_nonzero(bool1000)"),
    # --- other operations ---
    ("other", "np.linalg.det(3x3)", "np.linalg.det(m3)"),
    # --- creation / call overhead ---
    ("overhead", "np.array(1.5)", "np.array(1.5)"),
    ("overhead", "np.asarray(a)", "np.asarray(a)"),
    ("overhead", "np.zeros(10)", "np.zeros(10)"),
    ("overhead", "np.concatenate((a, b))", "np.concatenate((a, b))"),
    ("overhead", "np.result_type(a, 2.5)", "np.result_type(a, 2.5)"),
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
    parser.add_argument("--only", nargs="+", default=None, metavar="SUBSTR",
                        help="run only the cases whose name contains one of "
                             "these substrings")
    args = parser.parse_args()

    import numpy as np

    results = []
    cases = [c for c in CASES
             if args.only is None or any(sub in c[1] for sub in args.only)]
    for group, name, stmt in cases:
        ns = run_case(stmt)
        results.append({"group": group, "name": name, "ns": round(ns, 1)})
        print(f"  {group:>14} | {name:<24} {ns:9.1f} ns", file=sys.stderr)

    data = {
        "label": args.label or np.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "system": platform.system(),
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
