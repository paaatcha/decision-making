import numpy as np
from decision_making.topsis import TOPSIS

class ATOPSIS:

    def __init__(self, avg_mat, std_mat, weights=(0.75, 0.25), avg_cost_ben="benefit", std_cost_ben="cost",
                 alg_col_name=None, bench_col_names=None, bench_weights=None, normalize=False):
        """
        This class implements the A-TOPSIS algorithm (see README.md for references and citation).
        In the examples' folder you can see how to use this class. Obviously, this class depends on the TOPSIS one.

        Parameters:
        -----------
        avg_mat: (str, pd.DataFrame, [[]], np.array)
        This is the matrix that contains the average performance for the set of algorithms. It may be a string, a
        pd.DataFrame, a list of lists, or a numpy array. For all types except the string, it must be
        the decision matrix with its numerical values. In the DataFrame, you may also inform the algorithms and
        benchmarks names. If it's string, it must be the path to a .csv file to be loaded as a DataFrame

        std_mat: (str, pd.DataFrame, [[]], np.array)
        This is the matrix that contains the standard deviation performance for the set of algorithms. It may be a
        string, a pd.DataFrame, a list of lists, or a numpy array. For all types except the string, it must be
        the decision matrix with its numerical values. In the DataFrame, you may also inform the algorithms and
        benchmarks names. If it's string, it must be the path to a .csv file to be loaded as a DataFrame

        weights: (list, tuple), optional
        This is the weights you want to give to the average and standard deviation respectively. The default is
        (0.75, 0.25) as suggested by the original paper.

        avg_cost_ben: str, optional
        It defines the type of metric that the average matrix will assume. It may be a 'benefit' (or 'b') or 'cost'
        (or 'c') metric. For example, if the metric is accuracy, it's a benefit metric, since the higher, the better.
        On the other hand, if the metric is an error, it's cost because the lower, the better. Default is 'benefit'.

        std_cost_ben: str, optional
        It's similar to the avg_cost_ben, but for the standard deviation. Usually, it's 'cost', since we want a std
        as low as possible. Default is 'cost'

        alg_col_name: (str, None), optional
        This is a string with the name of the algorithm column. It must be informed if you're going to use the
        matrix_d as a DataFrame of a path to a csv. Check the examples and test to understand it better. If you're a
        list matrix or a numpy array, set it as None. Default is None.

        bench_col_names: (list, None), optional
        This is a list of strings containing the name of each benchmark column. It must be informed if you're going to
        use the matrix_d as a DataFrame of a path to a csv. Check the examples and test to understand it better. If
        you're a list matrix or a numpy array, set it as None. Default is None.

        bench_weights: (list, np.array, None), optional
        As the name suggest, it's the weights for each benchmark within the avg and std matrices. Obviously, it must
        have one weight for each benchmark. It's useful only if you have a benchmark that is more important than others.
        If you set it as None, it assumes that all benchmarks have the same weight. Regardless the option, the weights
        are always normalized within the interval [0,1]. Default is None.

        normalize: boolean, optional
        Set it as True if you want to normalize the avg and std matrices. Usually it's not necessary. Default is False.
        """

        self.avg_topsis = TOPSIS(avg_mat, weights=bench_weights, cost_ben=avg_cost_ben, alt_col_name=alg_col_name,
                                 crit_col_names=bench_col_names, normalize=normalize)
        self.std_topsis = TOPSIS(std_mat, weights=bench_weights, cost_ben=std_cost_ben, alt_col_name=alg_col_name,
                                 crit_col_names=bench_col_names, normalize=normalize)
        self.weights = list(weights)

        if not (self.avg_topsis.matrix_d.shape == self.std_topsis.matrix_d.shape):
            raise ValueError("The avg_mat and std_mat must have the same shape!")

        self.avg_ranking = None
        self.std_ranking = None
        self.matrix_d = None
        self.final_topsis = None
        self.final_ranking = None

    def get_avg_ranking(self):
        """
        This method computes the partial ranking for the avg metric. It saves the results in self.avg_topsis and
        self.avg_ranking.
        """
        self.avg_topsis.get_closeness_coefficient()
        self.avg_ranking = self.avg_topsis.clos_coefficient

    def get_std_ranking(self):
        """
        This method computes the partial ranking for the std metric. It saves the results in self.avg_topsis and
        self.avg_ranking.
        """
        self.std_topsis.get_closeness_coefficient()
        self.std_ranking = self.std_topsis.clos_coefficient

    def get_ranking(self, verbose=True):
        """
        This method computes the final raking taking into account the avg and std metrics. The results are saved in
        self.final_topsis and self.final_ranking

        Parameter:
        ---------
        verbose: boolean, optional
        Set it as True to show the results on the screen. Default is False
        """
        self.get_avg_ranking()
        self.get_std_ranking()
        self.matrix_d = np.array([self.avg_ranking, self.std_ranking]).T
        self.final_topsis = TOPSIS(self.matrix_d, weights=self.weights, cost_ben="b", normalize=False)
        self.final_topsis.get_closeness_coefficient(verbose)
        self.final_ranking = self.final_topsis.clos_coefficient

    def plot_ranking(self, alg_names=None, save_path=None, show=True, font_size=16, title="A-TOPSIS test", y_axis_title="Scores", x_axis_title="Methods", ascending=False, fig_size=(6,4)):
        """
        This method plots the ranking, according to the final_ranking, in a bar plot.

        Parameters:
        -----------
        alg_names: (list), optional
        This is a list of names for each algorithm within the metrics matrices. If you're using a DataFrame, you have
        already defined when you set the alg_col_name. So, you may let it as None. However, if you're using a matrix
        list or a numpy array, you may pass the algorithms name here. If you let it as None, there will be a default
        algorithms name in the plot (ex: A1, A2, etc). Default is None

        save_path: (str), optional
        It's the full path (including the figure name and extension) to save the plot. If you let it None, the plot
        won't be saved. Default is None.

        show: (boolean), optional
        Set is as True if you want to show the plot on the screen. Default is False.
        """
        if alg_names is None:
            alg_names = self.final_topsis.alternatives
        self.final_topsis.plot_ranking(alg_names, save_path, show, font_size, title, y_axis_title, x_axis_title, ascending=ascending, fig_size=fig_size)



















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

















