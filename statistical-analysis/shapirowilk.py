import numpy as np
from scipy import stats

def checkNormalization(data):
    statistic, p_value = stats.shapiro(data)
    alpha = 0.05

    if p_value > alpha:
        print(f"Normal data: p = {p_value:.3f}  , so p > {alpha}, value of W = {statistic}, data is likely  normal")
    else:
        print(f"Non-normal data: p = {p_value:.30f}  , so p < {alpha}, value of W = {statistic}, data is likely not normal")

def shapiroTest():
    normal_data = np.random.normal(loc=50, scale=5, size=100)
    print("--- Test 1 (Normal Data) ---")
    checkNormalization(normal_data)


    non_normal_data = np.random.uniform(low=0, high=100, size=100)
    print("\n--- Test 2 (Non-Normal Data) ---")
    checkNormalization(non_normal_data)
