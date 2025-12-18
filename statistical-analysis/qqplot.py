import matplotlib.pyplot as plt
import scipy.stats as stats


def generate_qq_plot_with_pvalue(data, plot_name):
    if len(data) < 3:
        print("Error: Not enough data points to calculate p-value.")
        return

    _, p_value = stats.shapiro(data)

    plt.figure(figsize=(10, 6))

    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm", plot=plt)

    text_str = f'Shapiro-Wilk Test\n$p$-value = {p_value:.5f}\n$R^2$ = {r**2:.4f}'

    plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.title(f'Q-Q Plot for {plot_name}', fontsize=16)
    plt.xlabel('Theoretical Quantiles', fontsize=12)
    plt.ylabel('Ordered Values', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.show()

    print(f"--- Statistics for {plot_name} ---")
    print(f"p-value: {p_value}")
    if p_value > 0.05:
        print("Result: Data looks Normal (fail to reject H0)")
    else:
        print("Result: Data does NOT look Normal (reject H0)")

def _plot_single_qq_on_ax(data, plot_name, ax):

    if len(data) < 3:
        print(f"Error for {plot_name}: Not enough data points to calculate p-value.")
        return

    _, p_value = stats.shapiro(data)

    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm", plot=ax)

    text_str = f'Shapiro-Wilk Test\n$p$-value = {p_value:.5f}\n$R^2$ = {r**2:.4f}'

    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=30,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    ax.set_title(f'Q-Q Plot for {plot_name}', fontsize=30)
    ax.set_xlabel('Theoretical Quantiles', fontsize=26)
    ax.set_ylabel('Ordered Values', fontsize=26)
    ax.grid(True, linestyle='--', alpha=0.6)

    print(f"--- Statistics for {plot_name} ---")
    print(f"p-value: {p_value}")
    if p_value > 0.05:
        print("Result: Data looks Normal (fail to reject H0)")
    else:
        print("Result: Data does NOT look Normal (reject H0)")

def generate_dual_qq_plots(data1, label1, data2, label2, filename='combined_qq_plots.png'):

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    _plot_single_qq_on_ax(data1, label1, axes[0])

    _plot_single_qq_on_ax(data2, label2, axes[1])

    plt.tight_layout()

    plt.savefig(filename)
    plt.close(fig)

    print(f"Combined Q-Q plot saved as '{filename}'")
