from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt


def _split_metrics_and_metadata(csv_path: Path) -> Tuple[str, Optional[int]]:
    """
    Return (metrics_csv_text, trailing_max_steps_or_None).

    If the last non-empty line starts with "max_steps", extract the integer and
    return the remaining lines *without* that metadata line. Otherwise, return
    the full file and *None*.
    """
    raw_lines = csv_path.read_text(encoding="utf-8").strip().splitlines()

    for i in range(len(raw_lines) - 1, -1, -1):
        line = raw_lines[i].strip()
        if not line:
            continue
        if line.lower().startswith("max_steps"):
            try:
                _, value_txt = line.split(",", 1)
                return "\n".join(raw_lines[:i]), int(value_txt)
            except ValueError as exc:
                raise ValueError(
                    f"Malformed max_steps line: '{line}'. Expected 'max_steps,<int>'."
                ) from exc
        break
    return "\n".join(raw_lines), None


def load_data(csv_path: Path) -> pd.DataFrame:
    """
    Load the CSV and return a tidy DataFrame ready for plotting.
    """

    metrics_text, trailing_max_steps = _split_metrics_and_metadata(csv_path)
    df = pd.read_csv(io.StringIO(metrics_text))

    required = {"generation", "best_fitness", "mean_fitness", "worst_fitness"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV file is missing the following required columns: {', '.join(sorted(missing))}"
        )

    if "max_steps" not in df.columns and trailing_max_steps is not None:
        df["max_steps"] = trailing_max_steps

    df = df.sort_values("generation").reset_index(drop=True)
    return df


def plot_fitness(df: pd.DataFrame, *, output: Path | None = None) -> None:
    """
    Plot fitness curves (and optional max_steps) and show or save the figure.
    """

    plt.figure(figsize=(10, 6))

    plt.plot(df["generation"], df["best_fitness"], label="Best fitness", linewidth=2)
    plt.plot(
        df["generation"],
        df["mean_fitness"],
        label="Mean fitness",
        linestyle="--",
        linewidth=2,
    )
    plt.plot(
        df["generation"],
        df["worst_fitness"],
        label="Worst fitness",
        linestyle=":",
        linewidth=2,
    )

    if "max_steps" in df.columns:
        plt.plot(
            df["generation"],
            df["max_steps"],
            color="black",
            linewidth=2,
            label="Max steps",
        )

    plt.xlabel("Generation")
    plt.ylabel("Fitness / Steps")

    title = "Evolution of Fitness Metrics"
    if "max_steps" in df.columns and df["max_steps"].nunique() == 1:
        title += f"  (max_steps = {int(df['max_steps'].iloc[0])})"
    plt.title(title)

    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    if output is not None:
        output = output.with_suffix(".png")
        plt.savefig(output, dpi=300)
        print(f"Plot saved to {output.resolve()}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot best, mean, and worst fitness over generations from a NEAT/GA "
            "CSV. If a per-generation max_steps column is present (or supplied "
            "via trailing metadata), it is overlaid as a solid black reference "
            "line."
        )
    )
    parser.add_argument("csv", type=Path, help="Path to the CSV file with fitness logs.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the plot as a PNG. If omitted, the plot opens in a window.",
    )
    args = parser.parse_args()

    df = load_data(args.csv)
    plot_fitness(df, output=args.output)


if __name__ == "__main__":
    main()
