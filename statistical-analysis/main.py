import pandas as pd
import numpy as np

import shapirowilk as sw
import qqplot as qqp
import mann_whitney as mw
import boxplot as bp
import os


def main():
    fitness_path = 'Fitness-Data'
    novelty_path = 'Novelty-Data'

    highest_fitness_list = get_highest_fitness_from_files(fitness_path)
    highest_nov_list = get_highest_fitness_from_files(novelty_path)

    print("\nList of highest 'best_fitness' values from all files:")
    print(highest_fitness_list)
    print(highest_nov_list)

    #fit_arr, nov_arr, p_value = mw.stats_on_all_files()
    data_sets = [highest_fitness_list, highest_nov_list]
    p_value = mw.stats_betweem_only_two(highest_fitness_list, highest_nov_list)
    labels = ['Fitness', 'Novelty']
    bp.create_boxplots(data_sets, labels, p_value)
    qqp.generate_dual_qq_plots(highest_fitness_list, "Fitness", highest_nov_list, "Novelty", "comb_qq_plot.png")
    #qqp.generate_qq_plot_with_pvalue(highest_fitness_list, "Fitness")
    #qqp.generate_qq_plot_with_pvalue(highest_nov_list, "Novelty")
    #mw.stats_betweem_only_two(mean_fitness_array, mean_nov_fitness_array)
    print("Application finished.")


def get_highest_fitness_from_files(fitness_path='Fitness-Data'):
    """
    Reads all CSV files in a specified directory, finds the highest
    value in the 'best_fitness' column of each file, and returns a list
    of these highest values.
    """

    # List to store the highest 'best_fitness' from each file
    highest_fitness_values = []

    # Check if the fitness_path exists
    if not os.path.exists(fitness_path):
        print(f"Error: The directory '{fitness_path}' does not exist.")
        return []

    # Get a list of all files in the directory
    all_fitness_files = os.listdir(fitness_path)

    # Filter for only CSV files
    csv_files = [f for f in all_fitness_files if f.endswith('.csv')]

    if not csv_files:
        print(f"No CSV files found in '{fitness_path}'.")
        return []

    print(f"Processing {len(csv_files)} files from '{fitness_path}'...")

    for file_name in csv_files:
        full_path = os.path.join(fitness_path, file_name)

        try:
            # Read the CSV file
            df = pd.read_csv(full_path)

            # Ensure the 'best_fitness' column exists
            if 'best_fitness' in df.columns:
                # Get the maximum value from the 'best_fitness' column
                highest_fitness = df['best_fitness'].max()

                # Add the highest number to the list
                highest_fitness_values.append(highest_fitness)

            else:
                print(f"Warning: 'best_fitness' column not found in {file_name}")

        except Exception as e:
            print(f"Error reading or processing file {file_name}: {e}")

    return highest_fitness_values


if __name__ == "__main__":
    main()
