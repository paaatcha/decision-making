import numpy as np
import pandas as pd
from decision_making.topsis import TOPSIS

class ATOPSIS:

    def __init__(self, avg_mat, std_mat, weights=(0.75, 0.25), avg_cost_ben="benefit", std_cost_ben="cost",
                 alg_col_name=None, bench_col_names=None, bench_weights=None):

        self.avg_topsis = TOPSIS(avg_mat, weights=bench_weights, cost_ben=avg_cost_ben, alt_col_name=alg_col_name,
                                 crit_col_names=bench_col_names)
        self.std_topsis = TOPSIS(std_mat, weights=bench_weights, cost_ben=std_cost_ben, alt_col_name=alg_col_name,
                                 crit_col_names=bench_col_names)
        self.weights = list(weights)

        if not (self.avg_topsis.matrix_d.shape == self.std_topsis.matrix_d.shape):
            raise ValueError("The avg_mat and std_mat must have the same shape!")

    def get_avg_ranking(self):
        self.avg_topsis.get_closeness_coefficient()
        self.avg_ranking = self.avg_topsis.clos_coefficient

    def get_std_ranking(self):
        self.std_topsis.get_closeness_coefficient()
        self.std_ranking = self.std_topsis.clos_coefficient

    def get_ranking(self, verbose=True):
        self.get_avg_ranking()
        self.get_std_ranking()
        self.matrix_d = np.array([self.avg_ranking, self.std_ranking]).T
        self.final_topsis = TOPSIS(self.matrix_d, weights=self.weights, cost_ben="b")
        self.final_ranking = self.final_topsis.get_closeness_coefficient(verbose)

    def plot_ranking(self, alt_names=None, save_path=None, show=True):
        self.final_topsis.plot_ranking(alt_names, save_path, show)



















# # The matrix of means
#
#
# #applying topsis to means
# #Tm = topsis ('valsMeans.txt') # You can also load it from a file
# Tm = TOPSIS(vals, w, cb)
# Tm.introWeights()
# Tm.getIdealSolutions()
# Tm.distanceToIdeal()
# Tm.relativeCloseness()
#
# #applying topsis to std
# #Ts = topsis ('valsStd.txt') # You can also load it from a file
# Ts = TOPSIS(stdVals, w, cb)
# Ts.introWeights()
# Ts.getIdealSolutions()
# Ts.distanceToIdeal()
# Ts.relativeCloseness()
#
# #applyting topsis for the rcloseness
# rcs = np.array ([Tm.rCloseness, Ts.rCloseness])
# rcs = rcs.T
#
# weightsFinal = np.array ([0.6, 0.4]) # (mean weight, std weigth)
# costBenFinal = np.array([0, 0]) #Benefit criteria
#
# Tf = TOPSIS(rcs, weightsFinal, costBenFinal)
# Tf.introWeights()
# Tf.getIdealSolutions()
# Tf.distanceToIdeal()
# Tf.relativeCloseness()
#
#
# print (Tf.rCloseness)
#
# Alternatives = np.array (['Alg1','Alg2','Alg3','Alg4'])
# Tf.plotRankBar(Alternatives)

















