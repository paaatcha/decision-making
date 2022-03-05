# import sys
# sys.path.append("../src")
# from decision_making import TODIM
#
# td = TODIM([[1,2], [1,2]])
# print(td.matrix_d)
#
# # -*- coding: utf-8 -*-
# """
# Author: Andre Pacheco
# Email: pacheco.comp@gmail.com
#
# An examples of how to use the topsis class.
#
# """
#
# from __future__ import print_function
# from __future__ import division
# from __future__ import absolute_import
#
# from src.decision_making.topsis import TOPSISimport sys
# sys.path.append("../src")
# from decision_making import TODIM
#
# td = TODIM([[1,2], [1,2]])
# print(td.matrix_d)
#
# # -*- coding: utf-8 -*-
# """
# Author: Andre Pacheco
# Email: pacheco.comp@gmail.com
#
# An examples of how to use the topsis class.
#
# """
#
# from __future__ import print_function
# from __future__ import division
# from __future__ import absolute_import
#
# from src.decision_making.topsis import TOPSIS
# import numpy as np
#
# A = TOPSIS('decisionMatrix.txt')
# A.normalizeMatrix()
# A.introWeights()
# A.getIdealSolutions()
# A.distanceToIdeal()
# A.relativeCloseness()
#
# # Showing the results
# print(A.rCloseness)
# Alternatives = np.array(['A1', 'A2', 'A3'])
# A.plotRankBar(Alternatives)
#
# # If you don't wanna use the file .txt, you can set the values
# # as lists or numpy arrays
#
# w = np.array([0.3, 0.05, 0.6, 0.05])
# cb = np.array([1, 0, 1, 0])
# matrix = np.array([
#     [15, 6, 25000, 7],
#     [12, 7, 35000, 7],
#     [10, 9, 55000, 8]
# ])
#
# B = TOPSIS(matrix, w, cb)
# B.normalizeMatrix()
# B.introWeights()
# B.getIdealSolutions()
# B.distanceToIdeal()
# B.relativeCloseness()
#
# # Showing the results
# print(B.rCloseness)
#
#
#
#
#
# import numpy as np
#
# A = TOPSIS('decisionMatrix.txt')
# A.normalizeMatrix()
# A.introWeights()
# A.getIdealSolutions()
# A.distanceToIdeal()
# A.relativeCloseness()
#
# # Showing the results
# print(A.rCloseness)
# Alternatives = np.array(['A1', 'A2', 'A3'])
# A.plotRankBar(Alternatives)
#
# # If you don't wanna use the file .txt, you can set the values
# # as lists or numpy arrays
#
# w = np.array([0.3, 0.05, 0.6, 0.05])
# cb = np.array([1, 0, 1, 0])
# matrix = np.array([
#     [15, 6, 25000, 7],
#     [12, 7, 35000, 7],
#     [10, 9, 55000, 8]
# ])
#
# B = TOPSIS(matrix, w, cb)
# B.normalizeMatrix()
# B.introWeights()
# B.getIdealSolutions()
# B.distanceToIdeal()
# B.relativeCloseness()
#
# # Showing the results
# print(B.rCloseness)
#
#
#
#
