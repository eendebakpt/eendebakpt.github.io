"""Plot benchmark results produced by run_benchmarks.py / bench_cases.py.

Reads all results/numpy-*.json files and writes PNG figures to images/:

    uv run --python 3.13 --no-project --with numpy,matplotlib plot_results.py

One line chart per benchmark group (time per call vs. numpy version) plus a
summary chart with the speedup of the newest vs. the oldest version.
"""

import json
import pathlib

import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent

# categorical palette (fixed slot order) and chart chrome
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

AXIS_TITLE_FONTSIZE = 13
TICK_FONTSIZE = 10.5

# operator/creation cases split off from the "scalar" group so no chart
# exceeds the 8-slot palette
SCALAR_MATH = ["f64 + f64", "f64 ** 2", "np.float64(1.5)", "i64 + i64"]

PLOTS = [
    ("scalar_ufunc", "Ufunc calls on scalars", "scalar",
     lambda name: name not in SCALAR_MATH),
    ("scalar_math", "Scalar math with operators", "scalar",
     lambda name: name in SCALAR_MATH),
    ("ufunc_small", "Ufuncs on a small array (n=10)", "ufunc-small", None),
    ("reduction_10", "Reductions on a small array (n=10)", "reduction-10",
     lambda name: name not in {"np.min", "np.all"}),
    ("reduction_1000", "Reductions on a contiguous array (n=1000)",
     "reduction-1000", lambda name: name not in {"np.min", "np.all"}),
    ("other_operations", "Other operations", ("nonzero", "other"), None),
    ("overhead", "Array creation and call overhead", "overhead", None),
]


def git_info(run):
    """Return (date, short_commit) for a git-derived numpy version, else None.

    Dev builds carry versions like 2.6.0.dev0+git20260811.60b906c.
    """
    parts = run["numpy"].split("+git")
    if len(parts) != 2 or "." not in parts[1]:
        return None
    date, commit = parts[1].split(".", 1)
    return date, commit[:7]


def version_key(run):
    """Sort releases by version and dev builds after them, by commit date."""
    info = git_info(run)
    if info is not None:
        # same-day commits: the later run (file mtime) is the newer commit
        return [99, int(info[0]), run.get("_mtime", 0)]
    try:
        return [int(p) for p in run["label"].split(".")[:3]]
    except ValueError:
        return [99, 0, 0]


def display_label(run, with_python=True):
    label = run["label"].split("+")[0].replace(".0.dev0", ".dev")
    info = git_info(run)
    if info is not None:
        date, short_commit = info
        base = ".".join(run["numpy"].split(".")[:2])
        # the run labelled "main-now" is the latest commit on the main branch
        name = "main" if run["label"] == "main-now" else f"{base}dev"
        label = f"{name}\n({date}, {short_commit})"
    if with_python:
        py = ".".join(run["python"].split(".")[:2])
        label += f"\npy{py}"
    return label


def load_results():
    """Load bench_cases results, one run per distinct numpy version.

    The runner writes a dev result twice (the "latest" file and a
    commit-specific copy), so runs are de-duplicated on the numpy version.
    """
    runs = {}
    for path in sorted((HERE / "results").glob("numpy-*.json")):
        with open(path) as fh:
            run = json.load(fh)
        if run["label"] == "2.6.0.dev0" or "results" not in run:
            continue
        if not run["results"] or "group" not in run["results"][0]:
            continue  # other benchmark formats, e.g. *-sin-sizes.json
        run["_mtime"] = path.stat().st_mtime
        runs.setdefault(run["numpy"], run)
    return sorted(runs.values(), key=version_key)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=TICK_FONTSIZE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_group(runs, group, title, outfile, name_filter=None):
    """Line chart of one benchmark group (or a tuple of groups) across versions."""
    groups = (group,) if isinstance(group, str) else tuple(group)
    labels = [display_label(r, with_python=False) for r in runs]
    names = [c["name"] for c in runs[-1]["results"] if c["group"] in groups]
    if name_filter is not None:
        names = [n for n in names if name_filter(n)]

    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    for tick, label in zip(ax.get_xticklabels(), labels):
        # dev builds carry a two-line "(date, commit)" label: rotate those
        if "\n" in label:
            tick.set_rotation(45)
            tick.set_ha("right")
            tick.set_fontsize(9)

    for idx, name in enumerate(names):
        values = []
        for run in runs:
            entry = {c["name"]: c["ns"] for c in run["results"]
                     if c["group"] in groups}
            values.append(entry.get(name))
        ax.plot(range(len(labels)), values, color=PALETTE[idx % len(PALETTE)],
                linewidth=2, marker="o", markersize=6, label=name)

    ax.set_ylim(bottom=0)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)
    ax.set_ylabel("time per call (ns)", color=INK2, fontsize=AXIS_TITLE_FONTSIZE)
    ax.set_xlabel("numpy version", color=INK2, fontsize=AXIS_TITLE_FONTSIZE)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2,
              loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(outfile, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {outfile}")


def plot_speedup(runs, outfile):
    """Horizontal bars: speedup of the newest vs. the oldest version.

    Compares only runs on the same CPython version, so the interpreter
    itself does not contribute to the measured speedup.
    """
    py = runs[-1]["python"].rsplit(".", 1)[0]
    same_py = [r for r in runs if r["python"].startswith(py)]
    old, new = same_py[0], same_py[-1]
    old_ns = {(c["group"], c["name"]): c["ns"] for c in old["results"]}
    new_ns = {(c["group"], c["name"]): c["ns"] for c in new["results"]}
    keys = [k for k in old_ns if k in new_ns]
    labels = [f"{name}  ({group})" for group, name in keys]
    speedups = [old_ns[k] / new_ns[k] for k in keys]

    order = sorted(range(len(keys)), key=lambda i: speedups[i])
    labels = [labels[i] for i in order]
    speedups = [speedups[i] for i in order]

    fig, ax = plt.subplots(figsize=(7.2, 0.28 * len(keys) + 1.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)

    ax.barh(labels, speedups, color="#2a78d6", height=0.62)
    ax.axvline(1.0, color=AXIS, linewidth=1)
    for i, s in enumerate(speedups):
        ax.text(s + 0.03, i, f"{s:.2f}x", va="center",
                color=INK2, fontsize=8)
    ax.set_xlim(0, max(speedups) * 1.15)
    old_name = display_label(old, with_python=False)
    new_name = display_label(new, with_python=False)
    ax.set_title(f"Speedup: numpy {old_name} → {new_name} "
                 "(higher is better)",
                 color=INK, fontsize=12, loc="left", pad=12)
    ax.set_xlabel("speedup factor", color=INK2, fontsize=AXIS_TITLE_FONTSIZE)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    fig.savefig(outfile, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {outfile}")


def main():
    runs = load_results()
    if not runs:
        raise SystemExit("no results found - run run_benchmarks.py first")
    (HERE / "images").mkdir(exist_ok=True)
    for fname, title, group, name_filter in PLOTS:
        plot_group(runs, group, title, HERE / "images" / f"{fname}.png",
                   name_filter)
    plot_speedup(runs, HERE / "images" / "speedup_summary.png")


if __name__ == "__main__":
    main()
