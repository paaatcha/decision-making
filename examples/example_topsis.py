"""
This is a straightforward example of how to use the TOPSIS algorithm with this package.
This example uses two approaches:
1: using the decision matrix inside a csv file
2: using a matrix in a list of lists
"""

import sys
sys.path.append("../src")
from decision_making import TOPSIS

dec_mat_2 = [
    [15, 6, 25000, 7],
    [12, 7, 35000, 7],
    [10, 9, 55000, 8]
]
alternatives = ["Alt 1", "Alt 2", "Alt 3"]
criteria = ["criterion 1", "criterion 2", "criterion 3", "criterion 4"]
weights = [0.3, 0.05, 0.6, 0.05]
cost_ben = ["c", "b", "c", "b"]

########################################################################################################################
# Approach 1: using the csv file in "../test/dec_mat_2.csv"
########################################################################################################################
print("-" * 50)
print("- Approach 1:")
print("-" * 50)
tp = TOPSIS("../test/dec_mat_2.csv", cost_ben, weights=weights, alt_col_name="alternative", crit_col_names=criteria)
tp.get_closeness_coefficient(verbose=True)
tp.plot_ranking()
print("-" * 50)
print("")

########################################################################################################################
# Approach 2: using the matrix as a list of list (but it could be a numpy array as well
########################################################################################################################
print("-" * 50)
print("- Approach 2:")
print("-" * 50)
tp = TOPSIS(dec_mat_2, cost_ben, weights=weights)
tp.get_closeness_coefficient(verbose=True)
tp.plot_ranking(alt_names=alternatives)
print("-" * 50)
print("")
