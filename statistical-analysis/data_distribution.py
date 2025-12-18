import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

novelty_dir = os.path.join("", "Novelty-Data")
fitness_dir = os.path.join("", "Fitness-Data")

THRESHOLDS = [100, 200, 300, 400, 500, 600, 700, 800, 900]


def load_all_histories(root_folder):

    histories = []

    for root, _, files in os.walk(root_folder):
        for name in files:
            lower = name.lower()
            if lower.startswith("fitness_history") and lower.endswith(".csv"):
                path = os.path.join(root, name)
                try:
                    df = pd.read_csv(path)
                except Exception as e:
                    print(f"Skipping unreadable CSV: {path} ({e})")
                    continue
                histories.append(df)
    return histories


def generation_hits_threshold(df, threshold):
    for _, row in df.iterrows():
        if row["best_fitness"] >= threshold:
            return int(row["generation"])
    return None


def analyze(histories):
    results = {}

    for t in THRESHOLDS:
        gens = []

        for df in histories:
            g = generation_hits_threshold(df, t)
            if g is not None:
                gens.append(g)

        if gens:
            print(f"Threshold {t} raw generations: {gens}")
            median_gen = int(np.median(gens))
            q1 = int(np.percentile(gens, 25))
            q3 = int(np.percentile(gens, 75))
            iqr = q3 - q1
            success_rate = len(gens) / len(histories) * 100
        else:
            median_gen = None
            iqr = None
            success_rate = 0

        results[t] = {
            "median_gen": median_gen,
            "iqr": iqr,
            "success_rate": success_rate
        }

    return results


def plot_results(novelty_results, fitness_results):
    thresholds = list(novelty_results.keys())

    novelty_medians = [
        novelty_results[t]["median_gen"] if novelty_results[t]["median_gen"] is not None else np.nan
        for t in thresholds
    ]

    fitness_medians = [
        fitness_results[t]["median_gen"] if fitness_results[t]["median_gen"] is not None else np.nan
        for t in thresholds
    ]

    plt.figure()
    plt.plot(thresholds, novelty_medians, marker='o', label="Novelty")
    plt.plot(thresholds, fitness_medians, marker='o', label="Fitness")

    plt.xlabel("Performance threshold")
    plt.ylabel("Median generation to reach threshold")
    plt.title("Generations required to reach performance thresholds")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("rq2_line_plot.png", dpi=300)
    plt.show()


def print_table(novelty_results, fitness_results):
    print("\nRQ2 TABLE DATA\n" + "-" * 40)
    print(f"{'Threshold':<12}{'Nov Med':<10}{'Nov IQR':<10}{'Nov %':<8}{'Fit Med':<10}{'Fit IQR':<10}{'Fit %'}")

    for t in THRESHOLDS:
        n = novelty_results[t]
        f = fitness_results[t]

        print(f"{t:<12}"
              f"{str(n['median_gen']):<10}"
              f"{str(n['iqr']):<10}"
              f"{n['success_rate']:.1f}%   "
              f"{str(f['median_gen']):<10}"
              f"{str(f['iqr']):<10}"
              f"{f['success_rate']:.1f}%")


if __name__ == "__main__":
    novelty_histories = load_all_histories(novelty_dir)
    fitness_histories = load_all_histories(fitness_dir)

    print(f"Loaded {len(novelty_histories)} novelty runs")
    print(f"Loaded {len(fitness_histories)} fitness runs")

    novelty_results = analyze(novelty_histories)
    fitness_results = analyze(fitness_histories)

    print_table(novelty_results, fitness_results)
    plot_results(novelty_results, fitness_results)
