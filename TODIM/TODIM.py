#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##############################################################
#      Author: Andre Pacheco (pacheco.comp@gmail.com)
#      To use this class you need to pass as argument to the constructor a file that contains the decision matrix, weights and theta's value
#      or just pass all these values as parameters.
#      For more information about TODIM:
#      [1] L.F.A.M. Gomes, M.M.P.P. Lima TODIM: Basics and application to multicriteria ranking of projects with environmental impacts
#      Foundations of Computing and Decision Sciences, 16 (4) (1992), pp. 113-127
#      Moreover, this code uses the chage proposed by Lourenzutti and Khroling (2013) when we compute the phi matrix.
#       Lourenzutti and Khroling (2013): : A study of TODIM in a intuitionistic fuzzy and random environment,
#       Expert Systems with Applications, Expert Systems with Applications 40 (2013) 6459-6468
#     If you use this code, please, cite:
#     [2] Krohling, Renato A., Andre GC Pacheco, and Andre LT Siviero. IF-TODIM: An intuitionistic fuzzy TODIM to multi-criteria decision making. Knowledge-Based Systems 53 #	(2013): 142-146.
#
#      If you find some bug, please e-mail me =)
#############################################################

import time
import numpy as np
import sys
sys.path.append('/home/patcha/Dropbox/Doutorado/Codigos/Python/Fuzzy Number')
import seaborn as sns
from matplotlib import pyplot as plt
from IntuitionisticFuzzyNumber import IntuitionisticFuzzyNumber


class TODIM:
    '''
    Attributes:
    matrixD - The decision matrix with the alternatives and criteria
    weights - The weights for each criteria
    theta - The theta's value
    nAlt - The number of alternatives
    nCri - The number of criteria
    normMatrixD - The matrixD normalized
    rCloseness - The relative closeness coeficient
    isFuzzy - flag for fuzzy number or not
    '''
    matrixD = None
    weights = None
    wref = None
    theta = None
    nAlt = None
    nCri = None
    normMatrixD = None
    phi = None
    delta = None
    rCloseness = None
    isFuzzy = False
    

    def __init__ (self, *args):
        nargs = len(args)        
        if nargs == 0:
            print 'ERROR: There is no parameter in the construction function'
            raise ValueError
        elif nargs == 1 or nargs == 2:
            # The .txt file need to be the 1st parameter
            fileName = args[0]
            try:
                data = np.loadtxt(fileName, dtype=float)
            except IOError:
                print 'ERROR: there is a problem with the .txt file. Please, check it again'
                raise IOError

            # All the values are in the .txt
            if nargs == 1:
                self.weights = data[0,:]
                self.theta = data[1,0]
                self.matrixD = data[2:,:]                
            # Only the matrixD is passed in the .txt
            else:
                self.matrixD = data
                self.weights = np.asarray(args[0])
                self.theta = args[1]
        # In this case, all the parameters are passed without use a .txt, in the following order: matrixD, weights, theta
        elif nargs == 3:
            self.matrixD = np.asarray(args[0])
            self.weights = np.asarray(args[1])
            self.theta = args[2]
               
        #Just checking if the weights' sum is equals 1
        if self.weights.sum() > 1.001 or self.weights.sum() < 0.9999:
            self.weights = self.weights/self.weights.sum()            
            print  'The weights was normalized in the interval [0,1]'            

        # Checking if the input is a Fuzzy Numbers
        if isinstance(self.matrixD[0,0], IntuitionisticFuzzyNumber):
            self.isFuzzy = True            
                        
        # Filling the remaining variables
        size = self.matrixD.shape
        [self.nAlt, self.nCri] = size
        self.normMatrixD = np.zeros(size, dtype=float)           
        self.delta = np.zeros([self.nAlt, self.nCri])
        self.rCloseness = np.zeros (self.nAlt, dtype=float)
        # weight reference
        self.wref = self.weights.max()

    def printTODIM (self):
        if self.isFuzzy:
            print 'MatrixD'
            for i in range(self.nAlt):
                print '\n'
                for j in range (self.nCri):
                    print i,j,'->',self.matrixD[i,j] 
        else:        
            print 'MatrixD \n', self.matrixD
        print 'Weights \n', self.weights
        print 'Theta \n', self.theta

    # Normalizeing the matrixD
    def normalizeMatrix (self):
        if self.isFuzzy:
            print 'TODO'
        else:
            m = self.matrixD.sum(axis=0)
#            m = self.matrixD ** 2
#            m = np.sqrt(m.sum(axis=0))
#    
            for i in range(self.nAlt):
                for j in range(self.nCri):
                    self.normMatrixD[i,j] = self.matrixD[i,j] / m[j]
    
            self.matrixD = self.normMatrixD

    # You can change the function, if you wanna do that. (Prepared to include fuzzy numbers)
    def distance (self, alt_i, alt_j, crit):
        if self.isFuzzy:
            return self.matrixD[alt_i, crit].distanceHamming(self.matrixD[alt_j, crit])
        else:
            return (self.matrixD[alt_i, crit] - self.matrixD[alt_j, crit])

    # If the value > 0, alt_i > alt_j, if = 0, they are equals, otherwise, alt_j > alt_i (prepared to include fuzzy numbers)
    def getComparison (self, alt_i, alt_j, crit):
        if self.isFuzzy:
            return self.matrixD[alt_i, crit].cmp(self.matrixD[alt_j, crit])
        else:
            return self.distance(alt_i, alt_j, crit)

    def getDelta (self):
        for i in range(self.nAlt):
            for j in range(self.nCri):
                self.delta[i,j] = self.getSumPhi(i,j)
                
    def getSumPhi (self,i,j):
        #m = np.zeros(self.nCri)
        m = 0
        for c in range(self.nCri):
            m = m + self.getPhi(i,j,c)
        return m
    
    def getPhi (self, i, j, c):
        dij = self.distance(i,j,c)
        comp = self.getComparison (i,j,c)
        if comp == 0:
            return 0
        elif comp > 0:
            return np.sqrt(self.weights[c]*abs(dij))
        else:
            return np.sqrt(self.weights[c]*abs(dij))/(-self.theta)

    def getRCloseness (self):
        self.getDelta()
        aux = self.delta.sum(axis=1)
        for i in range(self.nAlt):
            self.rCloseness[i] = (aux[i] - aux.min()) / (aux.max() - aux.min())
            
        print self.rCloseness
            

    def plotBars (self,names=None,save=None):
        sns.set_style("whitegrid")
        if names is not None:
            a = sns.barplot (names, self.rCloseness, palette="BuGn_d")
        else:
            a = sns.barplot (data=self.rCloseness, palette="BuGn_d")
        a.set_ylabel("Closeness Coeficient")
        a.set_xlabel('Alternatives')
        plt.show()
        fig = a.get_figure()
        
        if save is not None:
            fig.savefig(save+'.png')






################################## END CLASS ####################################################3


#mat = np.array([[IntuitionisticFuzzyNumber (), IntuitionisticFuzzyNumber ()],[IntuitionisticFuzzyNumber (),IntuitionisticFuzzyNumber ()]])
#y = IntuitionisticFuzzyNumber()

#t = TODIM (mat, [0.1, 0.9], 2.5)
#t.printTODIM()

#mat = np.array([[IntuitionisticFuzzyNumber ([0.5176, 0.5751, 0.6326]), IntuitionisticFuzzyNumber ([0.3134, 0.3482, 0.3830])],
#[IntuitionisticFuzzyNumber ([0.5903, 0.6559, 0.7215]), IntuitionisticFuzzyNumber ([0.2414, 0.2682, 0.2950])],
#[IntuitionisticFuzzyNumber ([0.6224, 0.6916, 0.7608]), IntuitionisticFuzzyNumber ([0.2097, 0.2330, 0.2563])],
#[IntuitionisticFuzzyNumber ([0.4920, 0.5467, 0.6013]), IntuitionisticFuzzyNumber ([0.3395, 0.3773, 0.4150])],
#[IntuitionisticFuzzyNumber ([0.3512, 0.3903, 0.4293]), IntuitionisticFuzzyNumber ([0.4793, 0.5326, 0.5859])],
#[IntuitionisticFuzzyNumber ([0.4865, 0.5405, 0.5946]), IntuitionisticFuzzyNumber ([0.3474, 0.3860, 0.4246])],
#[IntuitionisticFuzzyNumber ([0.4107, 0.4563, 0.5020]), IntuitionisticFuzzyNumber ([0.4250, 0.4722, 0.5194])],
#[IntuitionisticFuzzyNumber ([0.3443, 0.3825, 0.4208]), IntuitionisticFuzzyNumber ([0.4943, 0.5492, 0.6041])],
#[IntuitionisticFuzzyNumber ([0.3515, 0.3905, 0.4296]), IntuitionisticFuzzyNumber ([0.4913, 0.5459, 0.6005])],
#[IntuitionisticFuzzyNumber ([0.3761, 0.4179, 0.4597]), IntuitionisticFuzzyNumber ([0.4685, 0.5205, 0.5726])]])
#
#t = TODIM (mat, [0.5, 0.5], 1)
#t.getRCloseness()
#t.plotBars (np.array (['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10']))

t = TODIM ('exemplo3.txt')
t.normalizeMatrix()
t.getRCloseness()
t.plotBars(np.array (['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10', 'A11', 'A12', 'A13', 'A14', 'A15']))



