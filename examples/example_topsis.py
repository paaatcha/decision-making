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

# ideal_pos = [0.13852713, 0.03492677, 0.21483446, 0.03142697]
# ideal_neg = [0.20779069, 0.02328452, 0.47263582, 0.0274986]
# dist_pos = [0.07034498, 0.09070766, 0.25780135]
# dist_neg = [0.25780135, 0.17686323, 0.07034498]
# clos_coefficient = [0.78562925, 0.66099579, 0.21437075]


tp = TOPSIS(dec_mat_2, cost_ben, weights=weights)
# tp.apply_weights()
# tp.get_ideal_solutions()
# print(tp.ideal_neg)
# print(tp.ideal_pos)
# tp.get_distance_to_ideal()
tp.get_closeness_coefficient(verbose=True)