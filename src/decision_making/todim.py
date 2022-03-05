import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd


class TODIM:

    def __init__ (self, matrix_d, weights=None, theta=2.5, alt_col_name=None, crit_col_names=None, normalize=False):
        """
        This class implements the TODIM algorithm (see README.md for references and citation).
        In the examples' folder you can see how to use this class.

        Parameters:
        ----------
        matrix_d: (str, pd.DataFrame, [[]], np.array)
        This is the decision matrix. It may be a string, a pd.DataFrame, a list of lists, or a numpy array.
        For all types except the string, it must be the decision matrix with its numerical values. In the DataFrame,
        you may also inform the alternatives and criteria names. If it's string, it must be the path to a .csv file
        to be loaded as a DataFrame

        weights: (list, np.array, None)
        As the name suggest, it's the weights for each criterion within the decision matrix. Obviously, it must have one
        weight for each criterion. However, if you set it as None (the default value), it assumes that all criteria have
        the same weight. Regardless the option, the weights are always normalized.

        theta: float
        This the Theta value of the TODIM algorithm. Default is 2.5

        alt_col_name: (str, None)
        This is a string with the name of the alternative column. It must be informed if you're going to use the
        matrix_d as a DataFrame of a path to a csv. Check the examples and test to understand it better. If you're a
        list matrix or a numpy array, set it as None.

        crit_col_names: (list, None)
        This is a list of strings containing the name of each criterion column. It must be informed if you're going to
        use the matrix_d as a DataFrame of a path to a csv. Check the examples and test to understand it better. If
        you're a list matrix or a numpy array, set it as None.

        normalize: (boolean), optional
        Set it as True if you want to normalize the decision matrix. Default is False.
        """

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
        self.clos_coefficient = np.zeros([self.n_alt, 1], dtype=float)

        # Normalizing the decision matrix (if applicable)
        if normalize:
            self.normalizing_matrix_d()

    def print(self):
        """
        A simple method to print the decision matrix, weights, and theta
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
        print(f"- Theta: {self.theta}")
        print("-" * 50)

    def normalizing_matrix_d(self):
        """
        This method just normalizes the decision matrix within the interval [0,1] according to the values inside of the
        same criterion. The normalized matrix overwrites the self.matrix_d
        """
        crit_sum = self.matrix_d.sum(axis=0)
        self.matrix_d = self.matrix_d / crit_sum

    def get_distance(self, alt_i, alt_j, crit):
        """
        This method computes the distance between two alternatives for the same critetion. This is the standard way
        proposed by the TODIM method. However, there are some variants described in the literature. Thus, the
        get_distance and get_comparison are implemented in a separated way just in case you want to modify any of them
        without impacting the other (since there are variants that do it)

        Parameters:
        ----------
        alt_i: (int)
        An int indicating the position of the first alternative
        alt_j: (int)
        An int indication the position of the second alternative
        crit: (int)
        An int indicating the criterion in which we must take the alternatives

        Returns:
        -------
        _ : (float)
        The distance between the alternatives
        """
        return self.matrix_d[alt_i, crit] - self.matrix_d[alt_j, crit]

    def get_comparison(self, alt_i, alt_j, crit):
        """
        This method computes the distance between two alternatives for the same critetion. This is the standard way
        proposed by the TODIM method. However, there are some variants described in the literature. Thus, the
        get_distance and get_comparison are implemented in a separated way just in case you want to modify any of them
        without impacting the other (since there are variants that do it). For example, you may want to use a different
        distance and use this distance to create a different comparison method. It might be confusing, but to sum up:
        the distance and the comparison don't need to be the same. This is the reason to split them.

        Parameters:
        ----------
        alt_i: (int)
        An int indicating the position of the first alternative
        alt_j: (int)
        An int indication the position of the second alternative
        crit: (int)
        An int indicating the criterion in which we must take the alternatives

        Returns:
        -------
        _ : (float)
        The comparison between the alternatives
        """
        return self.get_distance(alt_i, alt_j, crit)

    def get_delta(self):
        """
        This method computes the Delta matrix as described in the TODIM algorithm. The result is saved in self.delta.
        """
        for i in range(self.n_alt):
            for j in range(self.n_crit):
                self.delta[i, j] = self.get_sum_phi(i,j)
                
    def get_sum_phi(self, i, j):
        """
        This method computes accumulated values of the Phi matrix as described in the TODIM algorithm.

        Parameters:
        -----------
        i: (int)
        An int with the alternative position
        j: (int)
        An int with the criterion position

        Returns:
        --------
        accum: (float)
        The accumulated value of the phi matrix according to the alternatives and criterion
        """
        accum = 0
        for c in range(self.n_crit):
            accum = accum + self.get_phi_matriz(i,j,c)
        return accum
    
    def get_phi_matriz(self, i, j, c):
        """
        This method computes the Phi matrix as described in the TODIM algorithm.

        Parameters:
        -----------
        i: (int)
        An int indicating the alternative position
        j: (int)
        An int indicating the alternative position
        c: (int)
        An int indicating the criterion position

        Returns:
        --------
        _: the phi matrix

        """
        dij = self.get_distance(i, j, c)
        comp = self.get_comparison(i, j, c)
        if comp == 0:
            return 0
        elif comp > 0:
            return np.sqrt(self.weights[c]*abs(dij))
        else:
            return np.sqrt(self.weights[c]*abs(dij))/(-self.theta)

    def get_closeness_coefficient(self, verbose=False):
        """
        This method uses the Delta matrix to compute the closeness coefficient, which the ranking computed by TODIM.
        The result is saved in self.closs_coefficient

        Parameters
        verbose: (boolean), optional
        Set is as true if you want to print the result on screen
        """
        self.get_delta()
        aux = self.delta.sum(axis=1)
        for i in range(self.n_alt):
            self.clos_coefficient[i] = (aux[i] - aux.min()) / (aux.max() - aux.min())
        self.clos_coefficient = self.clos_coefficient.squeeze()
        if verbose:
            print (self.clos_coefficient)

    def plot_ranking (self, alt_names=None, save_path=None, show=True):
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
            a = sns.barplot (alt_names, self.clos_coefficient, palette="BuGn_d")
        else:
            a = sns.barplot (None, self.clos_coefficient, palette="BuGn_d")
        a.set_ylabel("Closeness Coefficient")
        a.set_xlabel('Alternatives')
        fig = a.get_figure()

        if show:
            plt.show()
        
        if save_path is not None:
            fig.savefig(save_path)