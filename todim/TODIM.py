#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
Author: Andre Pacheco
Email: pacheco.comp@gmail.com

This class implements the todim [1,2] algorithm.
In order to use it, you need to inform the decision matrix, criteria's weights and thetas's value.
You can set these parameters in an external file .txt or just call the constructors passing 
the variables as parameters.

Moreover, this code uses the algoruthm's chage proposed [3] when we compute the phi matrix.

In the file task_todim.py there is an example showing how to use this class.

For more information about todim:
    [1] L.F.A.M. Gomes, M.M.P.P. Lima todim: Basics and application to multicriteria ranking of projects with environmental impacts
        Foundations of Computing and Decision Sciences, 16 (4) (1992), pp. 113-127
    
    [2] Krohling, Renato A., Andre GC Pacheco, and Andre LT Siviero. IF-todim: An intuitionistic fuzzy todim to multi-criteria decision
        making. Knowledge-Based Systems 53, (2013), pp. 142-146.

    [3] Lourenzutti, R. and Khroling, R. A study of todim in a intuitionistic fuzzy and random environment,
        Expert Systems with Applications, Expert Systems with Applications 40, (2013), pp. 6459-6468


If you find any bug, please e-mail me =)

'''

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd


class TODIM:

    def __init__ (self, matrix_d, weights=None, theta=2.5, alt_col_name=None, crit_col_names=None):
        '''
            Attributes:
            matrixD - The decision matrix with the alternatives and criteria. Shape = [# of alternatives x # of criteria]
            weights - The weights for each criteria
            theta - The theta's value
            nAlt - The number of alternatives
            nCri - The number of criteria
            normMatrixD - The matrixD normalized
            rCloseness - The relative closeness coeficient
        Args:
            matrix_d:
            weights:
            theta:
            alt_col_name:
            crit_col_names:
        '''


        # If the matrix_d is a string, we load it from a csv file
        if isinstance(matrix_d, str):
            matrix_d = pd.read_csv(matrix_d)

        self.criteria, self.alternatives = None, None

        # If the matrix_d is not a string, it may be a DataFrame, a list of lists or a Numpy array
        if isinstance(matrix_d, pd.DataFrame):
            # If it's a DataFrame, we need to check if alt_col_name and crit_col_names are filled
            if alt_col_name is not None:
                self.alternatives = matrix_d[alt_col_name].values
            if crit_col_names is not None:
                self.matrix_d = matrix_d[crit_col_names].values
                self.criteria = crit_col_names
            else:
                self.matrix_d = matrix_d.values
        elif isinstance(matrix_d, list):
            # If it's a list, we just transform it into a Numpy array
            self.matrix_d = np.asarray(matrix_d)
        elif isinstance(matrix_d, np.ndarray):
            # If it's a numpy array, we just get it
            self.matrix_d = matrix_d
        else:
            raise ValueError("The matrix_d parameter must be either a string, a DataFrame, a list of lists of a "
                             f"Numpy array. The type {type(matrix_d)} is not available at this moment.")

        # Getting the number of alternative and criteria
        self.n_alt, self.n_crit = self.matrix_d.shape

        if weights is None:
            self.weights = np.array([1] * self.n_crit) / self.n_crit
        else:
            if not isinstance(weights, list) or not isinstance(weights, np.ndarray):
                raise ValueError(f"The weights must be either a list or a Numpy array. The type {type(weights)} is "
                                 "not available at this moment.")
            self.weights = weights
            if self.weights.sum() > 1.001 or self.weights.sum() < 0.9999:
                self.weights = self.weights/self.weights.sum()
                print ("INFO: the weights were normalized within the interval [0,1]")

        self.w_ref = self.weights.max()
        self.theta = theta
        self.norm_matrix = np.zeros_like(self.matrix_d, dtype=float)
        self.delta = np.zeros_like(self.matrix_d, dtype=float)
        self.r_closeness = np.zeros([self.n_alt, 1], dtype=float)

        # # Filling the remaining variables
        # size = self.matrixD.shape
        # [self.nAlt, self.nCri] = size
        # self.normMatrixD = np.zeros(size, dtype=float)
        # self.delta = np.zeros([self.nAlt, self.nCri])
        # self.rCloseness = np.zeros ([self.nAlt,1], dtype=float)
        # # weight reference
        # self.wref = self.weights.max()

    def printTODIM (self):      
        print ('MatrixD \n', self.matrixD)
        print ('Weights \n', self.weights)
        print ('Theta \n', self.theta)

    # Normalizeing the matrixD
    def normalizeMatrix (self):
        m = self.matrixD.sum(axis=0)
        for i in range(self.nAlt):
            for j in range(self.nCri):
                self.normMatrixD[i,j] = self.matrixD[i,j] / m[j]
    
        self.matrixD = self.normMatrixD

    # You can change the function, if you wanna do that.
    def distance (self, alt_i, alt_j, crit):
        return (self.matrixD[alt_i, crit] - self.matrixD[alt_j, crit])
    
    # I use this function because it's easy to incluse another type of comparison
    def getComparison (self, alt_i, alt_j, crit):        
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

    def getRCloseness (self, verbose=False):
        self.getDelta()
        aux = self.delta.sum(axis=1)
        for i in range(self.nAlt):
            self.rCloseness[i] = (aux[i] - aux.min()) / (aux.max() - aux.min())
        if verbose:
            print (self.rCloseness)
            
    # To plot the Alternatives' name, just pass a list of names
    # To save the plot, just pass the files name on saveName
    def plotBars (self,names=None, saveName=None):        
        sns.set_style("whitegrid")
        if names is not None:
            a = sns.barplot (names, self.rCloseness[:,0], palette="BuGn_d")
        else:
            a = sns.barplot (None, self.rCloseness[:,0], palette="BuGn_d")
        
        a.set_ylabel("Closeness Coeficient")
        a.set_xlabel('Alternatives')
        fig = a.get_figure()
        plt.show()
        
        
        if saveName is not None:
            fig.savefig(saveName+'.png')

################################## END CLASS ####################################################

