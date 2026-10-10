#!/usr/bin/env python3
"""Chart the scaling benchmark: combined accuracy per model as the tool count grows.

Reads the four subset results that eval/benchmark.py writes (3, 5, 8 and 14
tools) and draws one grouped bar chart, the specialist against the three
baselines. Every bar is a number from those files; nothing is typed in.

Usage:
    python eval/plot_scaling.py                               # results/ to demos/scaling.png
    python eval/plot_scaling.py --results results --out demos/scaling.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUBSET_SIZES = [3, 5, 8, 14]
SPECIALIST = "edge-mcp-caller"
SPECIALIST_COLOR = "#b5451b"
BASELINE_COLORS = ["#7a8a99", "#a6b2bd", "#c9d1d8"]
TITLE = "Combined accuracy as the tool count grows: 270M specialist against prompt-based baselines"
SOURCE_LINE = "Source: results/benchmark_v03_{3,5,8,14}tool.json, 30 requests per tool, temperature 0."


def load_subsets(results_dir: Path) -> dict[int, dict[str, dict]]:
    """Load one benchmark file per subset size and check they describe the same models.

    Args:
        results_dir: Directory holding benchmark_v03_<n>tool.json for each subset size.

    Returns:
        Mapping from subset size to that file's per-model results.

    Raises:
        FileNotFoundError: If a subset file is missing.
        ValueError: If a file's subset_size disagrees with its name, or the
            model set differs between subsets.
    """
    subsets: dict[int, dict[str, dict]] = {}
    for size in SUBSET_SIZES:
        path = results_dir / f"benchmark_v03_{size}tool.json"
        if not path.exists():
            raise FileNotFoundError(f"missing subset file {path}; run eval/benchmark.py --subset all")
        data = json.loads(path.read_text())
        if data["subset_size"] != size:
            raise ValueError(f"{path.name} says subset_size {data['subset_size']}, expected {size}")
        subsets[size] = data["models"]
    model_names = list(subsets[SUBSET_SIZES[-1]].keys())
    for size, models in subsets.items():
        if list(models.keys()) != model_names:
            raise ValueError(f"model set differs in the {size}-tool file: {list(models.keys())}")
    return subsets


def plot_scaling(subsets: dict[int, dict[str, dict]], out: Path) -> None:
    """Draw combined accuracy per model per subset size and write the PNG.

    Args:
        subsets: Output of load_subsets.
        out: Path of the PNG to write; parent directories are created.
    """
    model_names = list(subsets[SUBSET_SIZES[-1]].keys())
    labels = {name: f"{name}: {subsets[SUBSET_SIZES[-1]][name]['description']}" for name in model_names}
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=120)
    width = 0.8 / len(model_names)
    baseline_index = 0
    for i, name in enumerate(model_names):
        accuracies = [100 * subsets[size][name]["combined_accuracy"] for size in SUBSET_SIZES]
        positions = [k + (i - (len(model_names) - 1) / 2) * width for k in range(len(SUBSET_SIZES))]
        if name == SPECIALIST:
            color = SPECIALIST_COLOR
        else:
            color = BASELINE_COLORS[min(baseline_index, len(BASELINE_COLORS) - 1)]
            baseline_index += 1
        bars = ax.bar(positions, accuracies, width=width * 0.95, color=color, label=labels[name], zorder=3)
        for bar, value in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.8, f"{value:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(SUBSET_SIZES)))
    ax.set_xticklabels([f"{size} tools" for size in SUBSET_SIZES])
    ax.set_ylabel("combined accuracy (tool and arguments), %")
    ax.set_ylim(0, 108)
    ax.grid(axis="y", color="#e3e6ea", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(TITLE, fontsize=11, pad=36)
    ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.01, SOURCE_LINE, fontsize=7, color="#666666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)


def main() -> None:
    """Parse arguments, load the subset files and write the chart."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"), help="directory with benchmark_v03_<n>tool.json")
    parser.add_argument("--out", type=Path, default=Path("demos/scaling.png"), help="PNG to write")
    args = parser.parse_args()
    subsets = load_subsets(args.results)
    plot_scaling(subsets, args.out)
    print(f"wrote {args.out} from {len(subsets)} subset files: {', '.join(subsets[SUBSET_SIZES[-1]].keys())}")


if __name__ == "__main__":
    main()
