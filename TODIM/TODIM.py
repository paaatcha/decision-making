#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
Author: Andre Pacheco
Email: pacheco.comp@gmail.com

This class implements the TODIM [1,2] algorithm.
In order to use it, you need to inform the decision matrix, criteria's weights and thetas's value.
You can set these parameters in an external file .txt or just call the constructors passing 
the variables as parameters.

Moreover, this code uses the algoruthm's chage proposed [3] when we compute the phi matrix.

In the file task_todim.py there is an example showing how to use this class.

For more information about TODIM:
    [1] L.F.A.M. Gomes, M.M.P.P. Lima TODIM: Basics and application to multicriteria ranking of projects with environmental impacts
        Foundations of Computing and Decision Sciences, 16 (4) (1992), pp. 113-127
    
    [2] Krohling, Renato A., Andre GC Pacheco, and Andre LT Siviero. IF-TODIM: An intuitionistic fuzzy TODIM to multi-criteria decision
        making. Knowledge-Based Systems 53, (2013), pp. 142-146.

    [3] Lourenzutti, R. and Khroling, R. A study of TODIM in a intuitionistic fuzzy and random environment,
        Expert Systems with Applications, Expert Systems with Applications 40, (2013), pp. 6459-6468


If you find any bug, please e-mail me =)

'''

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


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
    
    

    def __init__ (self, *args):
        nargs = len(args)        
        if nargs == 0:
            print ('ERROR: There is no parameter in the construction function')
            raise ValueError
        elif nargs == 1 or nargs == 2:
            # The .txt file need to be the 1st parameter
            fileName = args[0]
            try:
                data = np.loadtxt(fileName, dtype=float)
            except IOError:
                print ('ERROR: there is a problem with the .txt file. Please, check it again')
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
            print  ('The weights was normalized in the interval [0,1]')
            
                        
        # Filling the remaining variables
        size = self.matrixD.shape
        [self.nAlt, self.nCri] = size
        self.normMatrixD = np.zeros(size, dtype=float)           
        self.delta = np.zeros([self.nAlt, self.nCri])
        self.rCloseness = np.zeros ([self.nAlt,1], dtype=float)
        # weight reference
        self.wref = self.weights.max()

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

