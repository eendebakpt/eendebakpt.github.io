"""Run bench_cases.py against multiple numpy versions using uv.

Each version is executed in an isolated uv environment:

    python run_benchmarks.py                     # full matrix
    python run_benchmarks.py --versions 2.5.1    # subset

Python 3.13 is supported from numpy 2.1 onward, so older releases run on the
newest CPython they support (uv downloads the interpreter on demand).  Note
that per-call overhead depends on the CPython version as well, so numbers
measured on different interpreters are only indicative.

The development version is the nightly wheel from
https://anaconda.org/scientific-python-nightly-wheels.

Results are written to results/numpy-<version>.json and can be plotted with
plot_results.py.

A case added later can be measured on its own and merged into the existing
result files (matched on the numpy version):

    python run_benchmarks.py --only "np.linalg.det"
"""

import argparse
import os
import pathlib
import subprocess
import sys
import json
import tempfile

HERE = pathlib.Path(__file__).parent

NIGHTLY_INDEX = "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple"

# numpy version -> newest supported CPython
MATRIX = {
    "1.24.4": "3.11",
    "1.26.4": "3.12",
    "2.1.3": "3.13",
    "2.2.6": "3.13",
    "2.3.5": "3.13",
    "2.4.6": "3.13",
    "2.5.1": "3.13",
}


def merge_results(new_path: pathlib.Path) -> None:
    """Merge the cases in new_path into all result files of that numpy version."""
    with open(new_path) as fh:
        new = json.load(fh)
    targets = []
    for path in sorted((HERE / "results").glob("numpy-*.json")):
        with open(path) as fh:
            old = json.load(fh)
        if old.get("numpy") != new["numpy"] or not old.get("results") \
                or "group" not in old["results"][0]:
            continue
        index = {(c["group"], c["name"]): i for i, c in enumerate(old["results"])}
        for case in new["results"]:
            case = dict(case)
            if old.get("processor") != new.get("processor"):
                # the rest of this file was measured on another machine
                case["measured_on"] = f"{new.get('system', '')} {new['machine']}".strip()
            key = (case["group"], case["name"])
            if key in index:
                old["results"][index[key]] = case
            else:
                old["results"].append(case)
        with open(path, "w") as fh:
            fh.write(json.dumps(old, indent=2) + "\n")
        targets.append(path.name)
    if targets:
        print(f"    merged into {', '.join(targets)}", flush=True)
    else:
        print(f"    no result file for numpy {new['numpy']}, nothing merged",
              file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--versions", nargs="+", default=list(MATRIX),
                        help="numpy versions to benchmark")
    parser.add_argument("--specs", nargs="+", default=None,
                        help="package specs to benchmark instead of versions")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="labels for custom specs; only used with --specs")
    parser.add_argument("--python", default="3.13",
                        help="Python version to use for custom specs")
    parser.add_argument("--only", nargs="+", default=None, metavar="SUBSTR",
                        help="run only the matching cases and merge them into "
                             "the existing result files")
    args = parser.parse_args()
    if args.specs and args.labels and len(args.labels) != len(args.specs):
        parser.error("--labels must have the same number of values as --specs")

    (HERE / "results").mkdir(exist_ok=True)
    if args.specs:
        targets = list(zip(args.specs, args.labels or [None] * len(args.specs)))
    else:
        targets = [(version, None) for version in args.versions]

    for target, label in targets:
        if args.specs:
            spec = target
            sanitized = spec.replace(" ", "_").replace("/", "_")
            sanitized = sanitized.replace(":", "_").replace("=", "-")
            sanitized = sanitized.replace("@", "_").replace("+", "_")
            output_label = label or sanitized
            python = args.python
            output = HERE / "results" / f"numpy-{output_label}.json"
            print(f"=== {spec} (python {python}) ===", flush=True)
            cmd = [
                "uv", "run", "--python", python, "--no-project",
                "--with", spec,
                str(HERE / "bench_cases.py"), "--label", output_label,
                "--output", str(output)
            ]
        else:
            version = target
            python = MATRIX.get(version, "3.13")
            output = HERE / "results" / f"numpy-{version}.json"
            print(f"=== numpy {version} (python {python}) ===", flush=True)
            cmd = [
                "uv", "run", "--python", python, "--no-project",
                "--with", f"numpy=={version}",
            ]
            if "dev" in version:
                cmd += ["--index", NIGHTLY_INDEX,
                        "--index-strategy", "unsafe-best-match",
                        "--prerelease", "allow"]
            cmd += [str(HERE / "bench_cases.py"), "--output", str(output)]

        if args.only:
            # measure into a scratch file, then merge into the existing results
            scratch = pathlib.Path(tempfile.mkdtemp()) / "only.json"
            cmd[cmd.index("--output") + 1] = str(scratch)
            cmd += ["--only", *args.only]

        # PYTHONHOME/PYTHONPATH from the calling interpreter would leak the
        # wrong stdlib into the uv-managed interpreter
        env = {k: v for k, v in os.environ.items()
               if k not in ("PYTHONHOME", "PYTHONPATH")}
        result = subprocess.run(cmd, cwd=HERE, env=env)
        if result.returncode != 0:
            print(f"benchmark for {target} failed", file=sys.stderr)
            sys.exit(result.returncode)
        if args.only:
            merge_results(scratch)
            continue
        print(f"    -> {output}", flush=True)

        # If the run produced a dev wheel (git-derived) numpy version,
        # also write a commit-specific copy so older commit results are
        # preserved when the "latest" result is updated.
        try:
            if output.exists():
                with open(output, "r") as fh:
                    content = fh.read()
                data = json.loads(content)
                numpy_ver = data.get("numpy", "")
                if "+git" in numpy_ver:
                    parts = numpy_ver.split("+git")
                    if len(parts) == 2:
                        date_commit = parts[1]
                        if "." in date_commit:
                            date, commit = date_commit.split(".", 1)
                            # construct a human-friendly base like '2.6dev'
                            base_ver = parts[0]
                            bv_parts = base_ver.split(".")
                            if len(bv_parts) >= 2:
                                major = bv_parts[0]
                                minor = bv_parts[1]
                                commit_fname = f"numpy-{major}.{minor}dev-{date}-{commit[:7]}.json"
                            else:
                                commit_fname = f"numpy-{date}-{commit[:7]}.json"
                            commit_path = HERE / "results" / commit_fname
                            if not commit_path.exists():
                                with open(commit_path, "w") as fh:
                                    fh.write(content)
                                print(f"    -> {commit_path} (commit copy)", flush=True)
        except Exception:
            # Never fail the whole run for bookkeeping errors
            pass


if __name__ == "__main__":
    main()
