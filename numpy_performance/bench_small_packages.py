"""Compare NumPy with dedicated small-array packages on 10-element operations.

Times the same operations for numpy, tinyarray and lightarray and writes the
results as JSON plus a markdown table on stdout:

    taskset -c 2 uv run --python 3.13 --no-project \\
        --with "numpy @ git+https://github.com/numpy/numpy@<commit>" \\
        --with tinyarray --with lightarray \\
        bench_small_packages.py --output results/small-array-packages.json

A package that does not provide an operation gets no entry (tinyarray has no
trigonometric functions).
"""

import argparse
import json
import platform
import sys
import timeit

SETUP = """
import numpy as np
import tinyarray as ta
import lightarray as la

values = [float(i) for i in range(10)]
a = np.array(values); b = np.ones(10)
ta_a = ta.array(values); ta_b = ta.ones(10)
la_a = la.array(values); la_b = la.ones(10)
"""

# (row label, {package: statement})
CASES = [
    ("elementwise `a + b`",
     {"numpy": "a + b", "tinyarray": "ta_a + ta_b", "lightarray": "la_a + la_b"}),
    ("`np.sin(a)`",
     {"numpy": "np.sin(a)", "lightarray": "la.sin(la_a)"}),
    ("reduction (`np.sum` / `ta.dot` / `la.sum`)",
     {"numpy": "np.sum(a)", "tinyarray": "ta.dot(ta_a, ta_b)", "lightarray": "la.sum(la_a)"}),
    ("creation (`zeros(10)`)",
     {"numpy": "np.zeros(10)", "tinyarray": "ta.zeros(10)", "lightarray": "la.zeros(10)"}),
]
PACKAGES = ["numpy", "tinyarray", "lightarray"]

REPEAT = 9
TARGET_TIME = 0.02  # seconds per repeat


def run_case(stmt: str) -> float:
    """Return the best time per call in nanoseconds."""
    timer = timeit.Timer(stmt, setup=SETUP)
    number, elapsed = timer.autorange()
    number = max(10, int(number * TARGET_TIME / elapsed))
    return min(timer.repeat(repeat=REPEAT, number=number)) / number * 1e9


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    import numpy as np
    import tinyarray as ta
    import lightarray as la

    results = []
    for label, stmts in CASES:
        row = {"case": label}
        for package, stmt in stmts.items():
            row[package] = round(run_case(stmt), 1)
        results.append(row)

    versions = {"numpy": np.__version__, "tinyarray": ta.__version__,
                "lightarray": getattr(la, "__version__", "unknown")}
    print("| 10-element op | " + " | ".join(PACKAGES) + " |")
    print("|---|" + "---|" * len(PACKAGES))
    for row in results:
        cells = [f"{row[p]:.0f} ns" if p in row else "—" for p in PACKAGES]
        print(f"| {row['case']} | " + " | ".join(cells) + " |")
    print(versions, file=sys.stderr)

    data = {"versions": versions, "python": platform.python_version(),
            "system": platform.system(), "machine": platform.machine(),
            "results": results}
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
