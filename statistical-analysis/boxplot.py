import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def add_boxplot_to_ax(ax, data_set, i, label, color):
    pos = i + 1

    box = ax.boxplot([data_set],
                     patch_artist=True,
                     positions=[pos],
                     widths=0.6,
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(color='black'),
                     capprops=dict(color='black'),
                     showfliers=False)


    for patch in box['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    return box


def create_boxplots(data_sets, labels, p_val):
    print("\n--- Descriptive Statistics (Boxplot Components) ---")

    header = f"{'Dataset':<10} | {'Median':>8} | {'Q1':>6} | {'Q3':>6} | {'IQR':>6}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for label, data_set in zip(labels, data_sets):
        Q1 = np.percentile(data_set, 25)
        median = np.median(data_set)
        Q3 = np.percentile(data_set, 75)
        IQR = Q3 - Q1

        print(f"{label:<10} | {median:>8.2f} | {Q1:>6.2f} | {Q3:>6.2f} | {IQR:>6.2f}")

    print("-" * len(header))
    print("\n")

    _, ax = plt.subplots(figsize=(8, 6))
    all_boxes = []

    colors = sns.color_palette("Set2", len(data_sets))

    for i, (data_set, label) in enumerate(zip(data_sets, labels)):
        box = add_boxplot_to_ax(
            ax,
            data_set=data_set,
            i=i,
            label=label,
            color=colors[i]
        )
        all_boxes.append(box)


    all_values = np.concatenate(data_sets)
    best_overall = np.max(all_values)

    current_max_in_group = -np.inf
    best_group_pos = 1

    for i, data_set in enumerate(data_sets):
        if np.max(data_set) > current_max_in_group:
            current_max_in_group = np.max(data_set)
            best_group_pos = i + 1

    ax.scatter(best_group_pos, best_overall, color='red', s=250, marker='*', label='Best Overall', zorder=4)

    p_x_pos = (1 + len(data_sets)) / 2
    ax.text(p_x_pos, np.max(all_values) + 15, f"MW U p = {p_val:.2e}", ha='center', fontsize=18)

    ax.set_xticks(np.arange(len(data_sets)) + 1)
    ax.set_xticklabels(labels, fontsize=18)
    ax.set_ylabel("Top Performer Fitness", fontsize=18)
    ax.set_title("Top Performers per Method", fontsize=20)
    ax.legend()

    sns.despine(ax=ax)
    plt.show()
