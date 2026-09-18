"""Plot np.sin total time vs. array size, with a proportional fit for large arrays.

The fit is ``total_ns = slope * size`` (a line through the origin in normal
space) over all data points with ``size >= FIT_MIN_SIZE``, weighted by
``1 / total_ns`` (relative error) so every decade of sizes counts equally. On
log-log axes it is a straight line of slope 1 across the whole range. The
small arrays sit above it; the vertical gap is the per-call overhead, which is
reported as measured time at size 1 minus the fitted per-element cost.
"""

import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent

FIT_MIN_SIZE = 10**5

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


# The np.sin measurement is pinned to one NumPy build so the chart does not
# shift when new datapoints are added to the version charts. To re-measure,
# run bench_sin_sizes.py with this same build (see its docstring).
SIN_RESULTS = HERE / "results" / "numpy-2.6dev-20260918-e765e67-sin-sizes.json"


def load_sin_sizes_results():
    """Load the np.sin size benchmark results of the pinned NumPy build."""
    with open(SIN_RESULTS) as fh:
        return json.load(fh)


def numpy_version_label(version):
    """'2.6.0.dev0+git20260918.e765e67' -> '2.6.0.dev0, commit e765e67'."""
    base, _, git = version.partition("+git")
    if git and "." in git:
        return f"{base}, commit {git.split('.', 1)[1][:7]}"
    return version


def plot_sin_total_time():
    """Plot total time with a through-origin fit over sizes >= FIT_MIN_SIZE."""
    data = load_sin_sizes_results()
    results = data["results"]

    sizes = np.array([r["size"] for r in results], dtype=float)
    total_ns = np.array([r["total_ns"] for r in results], dtype=float)

    # Proportional fit in normal space: total_ns = slope * size (through the
    # origin). Weighted least squares with weights 1/total_ns (relative error)
    # so the largest arrays do not dominate the fit.
    large_mask = sizes >= FIT_MIN_SIZE
    x = sizes[large_mask] / total_ns[large_mask]
    slope_ns_per_elem = np.sum(x) / np.sum(x * x)
    fit_line = np.poly1d([slope_ns_per_elem, 0.0])
    # per-call overhead: what the smallest array costs beyond the fitted line
    overhead_ns = total_ns[0] - slope_ns_per_elem * sizes[0]

    # Create figure
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)

    ax.tick_params(colors=MUTED, labelsize=10.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    # Measured points
    ax.loglog(sizes, total_ns, "o", color="#2a78d6", markersize=5,
              label="np.sin total time (measured)", zorder=3)

    # Fitted line, drawn across the full range of sizes
    line_sizes = np.logspace(np.log10(sizes[0]), np.log10(sizes[-1]), 200)
    ax.loglog(line_sizes, fit_line(line_sizes), "--", color="#eb6834", linewidth=2.5,
              label=f"Linear fit through zero: {slope_ns_per_elem:.2f} ns/elem × size",
              zorder=2)

    # Mark the overhead gap at size 1
    ax.annotate("", xy=(sizes[0], total_ns[0]), xytext=(sizes[0], fit_line(sizes[0])),
                arrowprops=dict(arrowstyle="<->", color="#1baf7a", linewidth=1.5))
    # label in the wedge between the flat small-array points and the line
    ax.text(sizes[0] * 1.4, total_ns[0] * 0.5,
            f"per-call overhead\n≈ {overhead_ns:.0f} ns", color="#1baf7a",
            fontsize=9.5, va="center", ha="left")

    ax.set_xlabel("array size (log scale)", color=INK2, fontsize=13)
    ax.set_ylabel("total time (ns, log scale)", color=INK2, fontsize=13)
    ax.set_title(f"Computation time for np.sin (NumPy {numpy_version_label(data['numpy'])})",
                 color=INK, fontsize=12, loc="left", pad=14)

    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2,
              loc="upper left", bbox_to_anchor=(0.02, 0.98))

    fig.tight_layout()
    outfile = HERE / "images" / "sin_total_time.png"
    fig.savefig(outfile, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {outfile}")
    print(f"fit: slope={slope_ns_per_elem:.3f} ns/elem (through origin), "
          f"overhead at size 1={overhead_ns:.0f} ns, n={large_mask.sum()}")


if __name__ == "__main__":
    plot_sin_total_time()
