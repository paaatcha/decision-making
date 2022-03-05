import sys
sys.path.append("../src")
from decision_making import TODIM

td = TODIM([[1,2], [1,2]])
print(td.matrix_d)