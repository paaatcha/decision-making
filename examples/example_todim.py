"""
This is a straighforwarding example of how to use the TODIM algorithm with this package.
This example uses two approaches:
1: using the decision matrix inside a csv file
2: using a matrix in a list of lists
"""

import sys
sys.path.append("../src")
from decision_making import TODIM

weights = [0.5107, 0.4893]
theta = 2.5

########################################################################################################################
# Approach 1: using the csv file in "../test/dec_mat_1.csv"
########################################################################################################################

td_1 = TODIM ("../test/dec_mat_1.csv", weights=weights, theta=theta, alt_col_name="alternative",
              crit_col_names=["criterion 1", "criterion 2"])
print("-" * 50)
print("- Approach 1:")
print("-" * 50)
td_1.get_closeness_coefficient(verbose=True)
td_1.plot_ranking()
print("-" * 50)
print("")
########################################################################################################################
# Approach 2: using the matrix as a list of list (but it could be a numpy array as well
########################################################################################################################
dec_mat_1 = [
    [8.627, 5.223],
    [9.838, 4.023],
    [10.374, 3.495],
    [8.200, 5.659],
    [5.854, 7.989],
    [8.108, 5.790],
    [6.845, 7.083],
    [5.738, 8.238],
    [5.858, 8.189],
    [6.269, 7.808]
]

td_2 = TODIM (dec_mat_1, weights=weights, theta=theta)
print("-" * 50)
print("- Approach 2:")
print("-" * 50)
td_2.get_closeness_coefficient(verbose=True)
alternatives = ["Alg 1", "Alg 2", "Alg 3", "Alg 4", "Alg 5", "Alg 6", "Alg 7", "Alg 8", "Alg 9", "Alg 10"]
td_2.plot_ranking(alt_names=alternatives)
print("-" * 50)