import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

class TOPSIS:

      '''
      Attributes:
      matrixD - The decision matrix with the alternatives and criteria
      weights - The weights for each criteria
      costBen - A vector that represents if the criteria is a cost or benefit
      nAlt - The number of alternatives
      nCri - The number of criteria
      normMatrixD - The matrixD normalized
      idealPos and idealNeg - The ideal values positive and negative
      dPos and dNeg - The distance of each rating to the ideal value positive and negative
      rCloseness - The relative closeness coeficient
      '''

      def __init__ (self, matrix_d, cost_ben, weights=None, alt_col_name=None, crit_col_names=None):

            # If the matrix_d is a string, we load it from a csv file
            if isinstance(matrix_d, str):
                  matrix_d = pd.read_csv(matrix_d)

            self.criteria, self.alternatives = None, None

            # If the matrix_d is not a string, it must be either a DataFrame, a list of lists or a Numpy array
            if isinstance(matrix_d, pd.DataFrame):
                  # If it's a DataFrame, we need to check if alt_col_name and crit_col_names are filled
                  if alt_col_name is None or crit_col_names is None:
                        raise ValueError(
                              "You are using a DataFrame as input. Thus, you need to set the alt_col_name and "
                              "crit_col_names attributes")
                  self.alternatives = matrix_d[alt_col_name].values
                  self.matrix_d = matrix_d[crit_col_names].values
                  self.criteria = crit_col_names

            elif isinstance(matrix_d, list) or isinstance(matrix_d, np.ndarray):
                  # If it's a list or numpy array we just use it
                  self.matrix_d = np.asarray(matrix_d)
            else:
                  raise ValueError("The matrix_d parameter must be either a string, a DataFrame, a list of lists of a "
                                   f"Numpy array. The type {type(matrix_d)} is not available at this moment.")

            # Getting the number of alternative and criteria
            self.n_alt, self.n_crit = self.matrix_d.shape

            if weights is None:
                  self.weights = np.array([1] * self.n_crit) / self.n_crit
            else:
                  if not isinstance(weights, list) and not isinstance(weights, np.ndarray):
                        raise ValueError(
                              f"The weights must be either a list or a Numpy array. The type {type(weights)} is "
                              "not available at this moment.")
                  elif len(weights) != self.n_crit:
                        raise ValueError("The number of weights must be the same as the number of criteria")

                  self.weights = np.asarray(weights)
                  if not np.isclose(self.weights.sum(), 1.0):
                        self.weights = self.weights / self.weights.sum()
                        print("INFO: the weights were normalized within the interval [0,1]")

            self.cost_ben = cost_ben

            self.clos_coefficient = np.zeros([self.n_alt, 1], dtype=float)

            # size = self.matrixD.shape
            # [self.nAlt, self.nCri] = size
            # self.normMatrixD = np.zeros(size, dtype=float)
            # self.idealPos = np.zeros (self.nCri, dtype=float)
            # self.idealNeg = np.zeros (self.nCri, dtype=float)
            # self.dPos = np.zeros (self.nAlt, dtype=float)
            # self.dNeg = np.zeros (self.nAlt, dtype=float)
            # self.rCloseness = np.zeros (self.nAlt, dtype=float)

      def print(self):
            """
            A simple method to print the decision matrix, weights, and cost and benefit array
            """
            print("-" * 50)
            print("- Decision matrix:")
            print("-" * 50)
            print(self.matrix_d)

            print("-" * 50)
            print("- Weights:")
            print("-" * 50)
            print(self.weights)

            print("-" * 50)
            print(f"- Cost and benefit: {self.cost_ben}")
            print("-" * 50)


      def normalizing_matrix_d (self):
            m = self.matrix_d ** 2
            m = np.sqrt(m.sum(axis=0))

            for i in range(self.n_alt):
                  for j in range(self.n_crit):
                        self.matrix_d[i,j] = self.matrix_d[i,j] / m[j]

      def apply_weights (self):
            self.matrix_d = self.matrix_d * self.weights

      def getIdealSolutions (self):
            mx = self.normMatrixD.max(axis=0)
            mi = self.normMatrixD.min(axis=0)

            for j in range(self.nCri):
                  if self.costBen[j] == 1:
                        self.idealPos[j] =mi[j]
                        self.idealNeg[j] = mx[j]
                  elif self.costBen[j] == 0:
                        self.idealPos[j] = mx[j]
                        self.idealNeg[j] = mi[j]
                  else:
                        print ('ERROR: The values of the cost and benefit must be 1 or 0')
                        raise ValueError

      def distanceToIdeal (self):
            for i in range(self.nAlt):
                  for j in range(self.nCri):
                        self.dPos[i] = self.dPos[i] + distance (self.normMatrixD[i,j], self.idealPos[j])
                        self.dNeg[i] =self.dNeg[i] + distance (self.normMatrixD[i,j], self.idealNeg[j])

                  self.dPos[i] = np.sqrt(self.dPos[i])
                  self.dNeg[i] = np.sqrt(self.dNeg[i])

      def relativeCloseness (self):
            for i in range(self.nAlt):
                  self.rCloseness[i] = self.dNeg[i] / (self.dPos[i] + self.dNeg[i])


      # You need to pass as argument the name of the alternatives. If you wanna save the figure
      # Just pass the file name in saveName var.
      def plotRankBar (self, names, saveName=None):
            sns.set_style("whitegrid")
            a = sns.barplot (names, self.rCloseness, palette="Blues_d")
            a.set_ylabel("Closeness Coefficient")
            plt.show()

            if saveName is not None:
                  fig = a.get_figure()
                  fig.savefig(saveName+'.png', format='png')

      def plot_ranking(self, alt_names=None, save_path=None, show=True):
            """
            This method plots the ranking, according to the closeness coefficient, in a bar plot.

            Parameters:
            -----------
            alt_names: (list), optional
            This is a list of names for each alternative within the decision matrix. If you're using a DataFrame, you have
            already defined when you set the alt_col_name. So, you may let it as None. However, if you're using a matrix
            list or a numpy array, you may pass the alternatives name here. If you let it as None, there will be no
            alternatives name in the plot. Default is None

            save_path: (str), optional
            It's the full path (including the figure name and extension) to save the plot. If you let it None, the plot
            won't be saved. Default is None.

            show: (boolean), optional
            Set is as True if you want to show the plot on the screen. Default is False.
            """

            sns.set_style("whitegrid")
            if self.alternatives is not None:
                  alt_names = self.alternatives
            if alt_names is not None:
                  a = sns.barplot(alt_names, self.clos_coefficient[:, 0], palette="BuGn_d")
            else:
                  a = sns.barplot(None, self.clos_coefficient[:, 0], palette="BuGn_d")
            a.set_ylabel("Closeness Coefficient")
            a.set_xlabel('Alternatives')
            fig = a.get_figure()

            if show:
                  plt.show()

            if save_path is not None:
                  fig.savefig(save_path)


############################## END CLASS ###########################################

def distance (a,b):
      return (a-b)**2









