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

In the file task_todim.py there is an examples showing how to use this class.

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

        # If the matrix_d is not a string, it must be either a DataFrame, a list of lists or a Numpy array
        if isinstance(matrix_d, pd.DataFrame):
            # If it's a DataFrame, we need to check if alt_col_name and crit_col_names are filled
            if alt_col_name is None or crit_col_names is None:
                raise ValueError("You are using a DataFrame as input. Thus, you need to set the alt_col_name and "
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
                raise ValueError(f"The weights must be either a list or a Numpy array. The type {type(weights)} is "
                                 "not available at this moment.")
            elif len(weights) != self.n_crit:
                raise ValueError("The number of weights must be the same as the number of criteria")

            self.weights = np.asarray(weights)
            if not np.isclose(self.weights.sum(), 1.0):
                self.weights = self.weights/self.weights.sum()
                print ("INFO: the weights were normalized within the interval [0,1]")

        self.theta = theta
        self.delta = np.zeros_like(self.matrix_d, dtype=float)
        self.closs_coefficient = np.zeros([self.n_alt, 1], dtype=float)


    def print(self):
        print("-" * 50)
        print("- Decision matrix:")
        print("-" * 50)
        print(self.matrix_d)

        print("-" * 50)
        print("- Weights:")
        print("-" * 50)
        print(self.weights)

        print("-" * 50)
        print(f"- Theta: {self.theta}")
        print("-" * 50)

    def normalizing_matrix_d(self):
        """
        This method just normalizes the criteria of matrix_d within the interval [0,1]
        Returns:

        """
        crit_sum = self.matrix_d.sum(axis=0)
        for i in range(self.n_alt):
            for j in range(self.n_crit):
                self.matrix_d[i,j] = self.matrix_d[i,j] / crit_sum[j]

    def get_distance(self, alt_i, alt_j, crit):
        """
        This method computes the distance between two criteria. You may change it if you
        want a different distance function
        Args:
            alt_i:
            alt_j:
            crit:

        Returns:

        """
        return self.matrix_d[alt_i, crit] - self.matrix_d[alt_j, crit]
    
    # I use this function because it's easy to incluse another type of comparison
    def get_comparison (self, alt_i, alt_j, crit):
        return self.get_distance(alt_i, alt_j, crit)

    def get_delta (self):
        for i in range(self.n_alt):
            for j in range(self.n_crit):
                self.delta[i, j] = self.get_sum_phi(i,j)
                
    def get_sum_phi (self, i, j):
        accum = 0
        for c in range(self.n_crit):
            accum = accum + self.get_phi_matriz(i,j,c)
        return accum
    
    def get_phi_matriz (self, i, j, c):
        dij = self.get_distance(i, j, c)
        comp = self.get_comparison(i, j, c)
        if comp == 0:
            return 0
        elif comp > 0:
            return np.sqrt(self.weights[c]*abs(dij))
        else:
            return np.sqrt(self.weights[c]*abs(dij))/(-self.theta)

    def get_closeness_coefficient (self, verbose=False):
        self.get_delta()
        aux = self.delta.sum(axis=1)
        for i in range(self.n_alt):
            self.closs_coefficient[i] = (aux[i] - aux.min()) / (aux.max() - aux.min())
        if verbose:
            print (self.closs_coefficient)

    def plot_bars (self, alt_names=None, save_path=None):
        """
            # To plot the Alternatives' name, just pass a list of names
            # To save the plot, just pass the files name on saveName
        Args:
            alt_names:
            save_path:

        Returns:

        """
        sns.set_style("whitegrid")
        if self.alternatives is not None:
            alt_names = self.alternatives
        if alt_names is not None:
            a = sns.barplot (alt_names, self.closs_coefficient[:, 0], palette="BuGn_d")
        else:
            a = sns.barplot (None, self.closs_coefficient[:, 0], palette="BuGn_d")
        a.set_ylabel("Closeness Coefficient")
        a.set_xlabel('Alternatives')
        fig = a.get_figure()
        plt.show()
        
        if save_path is not None:
            fig.savefig(save_path)