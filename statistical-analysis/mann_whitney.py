import pandas as pd
import os
from scipy.stats import mannwhitneyu as mwu


def stats_on_all_files():
    fitness_path = 'Fitness-Data'
    novelty_path = 'Novelty-Data'

    all_fitness_files = os.listdir(fitness_path)
    all_novelty_files = os.listdir(novelty_path)

    fitness_collected_data = combine_files(all_fitness_files, fitness_path).copy()
    novelty_collected_data = combine_files(all_novelty_files, novelty_path).copy()


    mean_fitness_data = mean_of_each_gen(fitness_collected_data,
                                         get_smallest_file_size(all_fitness_files, fitness_path),
                                         len(all_fitness_files)).copy()
    mean_nov_data = mean_of_each_gen(novelty_collected_data,
                                     get_smallest_file_size(all_novelty_files, novelty_path),
                                     len(all_novelty_files))

    print(mean_fitness_data)
    print(mean_nov_data)
    p_value = stats_betweem_only_two(mean_fitness_data, mean_nov_data)
    return mean_fitness_data, mean_nov_data, p_value

def count_lines(filepath):
    with open(filepath, 'rb') as f:
        return sum(1 for line in f)

def get_largest_file_size(files: list, file_path):
    largest_file_size = 0

    for file in files:
        full_path = os.path.join(file_path, file)
        if not os.path.isfile(full_path):
            continue
        file_size = count_lines(full_path)

        if file_size > largest_file_size:
            largest_file_size = file_size

    return largest_file_size

def get_smallest_file_size(files: list, file_path):
    smallest_file_size = 500

    for file in files:
        full_path = os.path.join(file_path, file)
        if not os.path.isfile(full_path):
            continue
        file_size = count_lines(full_path)

        if file_size < smallest_file_size:
            smallest_file_size = file_size

    return smallest_file_size


def combine_files(files: list, file_path):
    combined_data = [0] * get_largest_file_size(files, file_path)
    for file in files:
        full_path = os.path.join(file_path, file)
        df = pd.read_csv(full_path)
        fitness_data = df['best_fitness'].to_numpy()
        for i in range(len(fitness_data)): #For adding all gen 0 with gen 0, and gen 1 with gen 1, and so on
            combined_data[i] += float(fitness_data[i])

    return combined_data

def mean_of_each_gen(data, range_size, num_of_datasets):
    mean_data = [0] * range_size
    for i in range(len(mean_data)):
        mean_data[i] = data[i]/num_of_datasets

    return mean_data


def stats_betweem_only_two(data1, data2):
    stat, p_value = mwu(data1, data2, alternative="two-sided")
    print(f"P-value of Mann-Whitney test: {p_value:.60f}, and the stat is: {stat:.60f}")
    return p_value
