"""
This is a straightforward example of how to use the A-TOPSIS algorithm with this package.
This example uses two approaches:
1: using the decision matrices inside a csv files
2: using the matrices in a list of lists

Important: in this example we use the avg_cost_ben="cost" because the performance metric is the error (the lower, the
better). If it was the accuracy, for example, we'd use it as "benefit", since for this metric the higher, the better.
You must double-check it before running the method.
"""

import sys
sys.path.append("../src")
import numpy as np
from decision_making import ATOPSIS


########################################################################################################################
# Approach 1: using the csv files in "../4test/avg_mat.csv" and in "../test/std_mat.csv"
########################################################################################################################
print("-" * 50)
print("- Approach 1:")
print("-" * 50)
atop = ATOPSIS("../test/avg_mat.csv", "../test/std_mat.csv", alg_col_name="Algorithms",
               avg_cost_ben="cost", std_cost_ben="cost", weights=[0.6, 0.4],
               bench_col_names=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"])

atop.get_ranking(True)
atop.plot_ranking()
print("-" * 50)
print("")
########################################################################################################################
# Approach 2: using the matrices as list of lists
########################################################################################################################
avg_mat = [
    [0.00349, 0.00053, 0.00171, 0.00972, 0.0309, 0.06638, 0.0489, 0.0246, 0.05088, 0.10506, 0.04228],
    [0.00273, 0.00056, 0.00142, 0.00922, 0.0298, 0.07131, 0.05402, 0.0241, 0.05011, 0.10275, 0.06492],
    [0.00002, 0.00008, 0.0003, 0.00222, 0.0657, 0.05182, 0.03953, 0.0312, 0.03469, 0.03853, 0.02752],
    [0.00009, 0.00013, 0.00071, 0.00534, 0.0263, 0.04387, 0.03946, 0.0222, 0.03459, 0.03842, 0.02749]
]
std_mat = [
    [0.00186, 0.00026, 0.00078, 0.00291, 0.00517, 0.03292, 0.01768, 0.00195, 0.01393, 0.06068, 0.02847],
    [0.00417, 0.00025, 0.00055, 0.00364, 0.00499, 0.02353, 0.00982, 0.00217, 0.01712, 0.07357, 0.03455],
    [0.00001, 0.00002, 0.00012, 0.0006, 0.0006, 0.02639, 0.00015, 0.00019, 0.00122, 0.00014, 0.00018],
    [0.00004, 0.00004, 0.00041, 0.00148, 0.00063, 0.01757, 0.00022, 0.00013, 0.00083, 0.00012, 0.0003]
]

print("-" * 50)
print("- Approach 2:")
print("-" * 50)
atop = ATOPSIS(avg_mat, std_mat, avg_cost_ben="cost", std_cost_ben="cost", weights=[0.6, 0.4])
atop.get_ranking(True)
atop.plot_ranking(alg_names=["Alg1", "Alg2", "Alg3", "Alg4"])
print("-" * 50)
print("")


