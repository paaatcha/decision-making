"""
This is a straightforward example of how to use the A-TOPSIS algorithm with this package.
This example uses two approaches:
1: using the decision matrices inside a csv files
2: using the matrices in a list of lists
"""

import sys
sys.path.append("../src")
import numpy as np
from decision_making import ATOPSIS


atop = ATOPSIS("../test/avg_mat.csv", "../test/std_mat.csv", alg_col_name="Algorithms",
               avg_cost_ben="cost", std_cost_ben="cost",
               bench_col_names=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"])

atop.get_ranking(True)



