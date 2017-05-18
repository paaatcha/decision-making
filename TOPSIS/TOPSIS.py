#      Author: Andre Pacheco (pacheco.comp@gmail.com)
#      To use this class you need to pass as argument to the constructor a file that contains the decision matrix, weights and cost/benefit information.
#      or just pass all these values as parameters
#      For more information about TOPSIS:
#      [1] C.L. Hwang & K.P. Yoon, Multiple Attributes Decision Making Methods and Applications, Springer-Verlag, Berlin, 1981.
#
#     If you use this code, please, cite:
#     [2] Krohling, Renato A., Andre GC Pacheco, and Andre LT Siviero. IF-TODIM: An intuitionistic fuzzy TODIM to multi-criteria decision making. Knowledge-Based Systems 53 #	(2013): 142-146.
#
#      If you find some bug, please e-mail me =)
#############################################################


import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

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

      def __init__ (self,*args):
            nargs = len(args)

            if nargs == 0:
                  print 'ERROR: There is no parameter'
                  raise ValueError
            elif nargs == 1 or nargs == 2:
                  # In this case we set all the parameters as a file or set only the matrixD and pass the weights and costBen via code
                  fileName = args[0]
                  try:
                        data = np.loadtxt(fileName, dtype=float)
                  except IOError:
                        print 'ERROR: there is a problem with the file. Please, check the name.'
			raise IOError

                  if nargs == 1:
                        self.weights = data[0,:]
                        if self.weights.sum() != 1.0:
                              print 'ERROR: the sum of the weights must be 1'
                              raise ValueError

                        self.costBen = data[1,:].astype(int)
                        self.matrixD = data[2:,:]
                  else:
                        self.matrixD = data
                        self.weights = args[0]
                        if self.weights.sum() != 1.0:
                              print 'ERROR: the sum of the weights must be 1'
                              raise ValueError
                        self.costBen = arg[1]

            elif nargs == 3:
                  # In this case, the parameters' order are: matrixD, weights and costBen
                  self.matrixD = args[0]
                  self.weights = args[1]
                  if self.weights.sum() != 1.0:
                              print 'ERROR: the sum of the weights must be 1'
                              raise ValueError
                  self.costBen = args[2]

            else:
                  print 'ERROR: The number of the parameters is wrong'
                  raise ValueError


            size = self.matrixD.shape
            [self.nAlt, self.nCri] = size
            self.normMatrixD = np.zeros(size, dtype=float)
            self.idealPos = np.zeros (self.nCri, dtype=float)
            self.idealNeg = np.zeros (self.nCri, dtype=float)
            self.dPos = np.zeros (self.nAlt, dtype=float)
            self.dNeg = np.zeros (self.nAlt, dtype=float)
            self.rCloseness = np.zeros (self.nAlt, dtype=float)


      def normalizeMatrix (self):
            m = self.matrixD **2
            m = np.sqrt(m.sum(axis=0))

            for i in range(self.nAlt):
                  for j in range(self.nCri):
                        self.normMatrixD[i,j] = self.matrixD[i,j] / m[j]

            self.matrixD = self.normMatrixD

      def introWeights (self):
            self.normMatrixD = self.matrixD * self.weights

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
                        print 'ERROR: The values of the cost and benefit must be 1 or 0'
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


      # You need to pass as argument the name of the alternatives
      def plotRankBar (self, names):
            sns.set_style("whitegrid")
            a = sns.barplot (names, self.rCloseness)
            a.set_ylabel("Ranking")
            plt.show()


############################## END CLASS ###########################################

def distance (a,b):
      return (a-b)**2


# HOW TO USE
A = TOPSIS ('mat_dec_votos.txt')
A.normalizeMatrix()
A.introWeights()
A.getIdealSolutions()
A.distanceToIdeal()
A.relativeCloseness()
Alternatives = np.array (['CRI','PEP','JAN','BOL','FRE','IDC','ALE','OSO','CYR','THE','CAR'])
A.plotRankBar(Alternatives)






