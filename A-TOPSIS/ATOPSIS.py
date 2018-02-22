'''
Author: Andre Pacheco
Email: pacheco.comp@gmail.com

This script implements the A-TOPSIS [1,2]. In order to use it, first you need to set the
path to the TOPSIS.py (also included in this repository). Next, you need to define the
the means and std matrices, as I did below. You also can get these matrices from a file.

For more information about A-TOPSIS:

[1] Krohling, R. A., and Pacheco, A.G.C. A-TOPSIS - an approach based on TOPSIS for ranking 
    evolutionary algorithms. Procedia Computer Science 55 (2015): 308-317.

[2] Pacheco, A.G.C. and Krohling, R.A. "Ranking of Classification Algorithms in Terms of 
    Mean-Standard Deviation Using A-TOPSIS". Annals of Data Science (2016), pp.1-18.

If you find some bug, please e-mail me =)

'''

# Setting the path to the TOPSIS class
import sys
sys.path.insert(0, '../TOPSIS')

from TOPSIS import TOPSIS
import numpy as np

# The matrix of means
vals = np.array( [[0.00349, 0.00273, 0.00002, 0.00009],
        [0.00053, 0.00056, 0.00008, 0.00013],
        [0.00171, 0.00142, 0.00030, 0.00071],
        [0.00972, 0.00922, 0.00222, 0.00534],
        [0.03090, 0.02980, 0.06570, 0.02630],
        [0.06638, 0.07131, 0.05182, 0.04387],
        [0.04890, 0.05402, 0.03953, 0.03946],
        [0.02460, 0.02410, 0.03120, 0.02220],
        [0.05088, 0.05011, 0.03469, 0.03459],
        [0.10506, 0.10275, 0.03853, 0.03842],
        [0.04228, 0.06492, 0.02752, 0.02749]] ).T

# The matrix of std
stdVals = np.array( [[0.00186, 0.00417, 0.00001, 0.00004],
           [0.00026, 0.00025, 0.00002, 0.00004], 
           [0.00078, 0.00055, 0.00012, 0.00041], 
           [0.00291, 0.00364, 0.00060, 0.00148],
           [0.00517, 0.00499, 0.00060, 0.00063],
           [0.03292, 0.02353, 0.02639, 0.01757],
           [0.01768, 0.00982, 0.00015, 0.00022],
           [0.00195, 0.00217, 0.00019, 0.00013],
           [0.01393, 0.01712, 0.00122, 0.00083],
           [0.06068, 0.07357, 0.00014, 0.00012],
           [0.02847, 0.03455, 0.00018, 0.00030]] ).T

w = np.ones([1,11]) / 11
cb = np.ones([11])

#applying TOPSIS to means
#Tm = TOPSIS ('valsMeans.txt') # You can also load it from a file
Tm = TOPSIS (vals, w, cb)
Tm.introWeights()
Tm.getIdealSolutions()
Tm.distanceToIdeal()
Tm.relativeCloseness()

#applying TOPSIS to std
#Ts = TOPSIS ('valsStd.txt') # You can also load it from a file
Ts = TOPSIS (stdVals, w, cb)
Ts.introWeights()
Ts.getIdealSolutions()
Ts.distanceToIdeal()
Ts.relativeCloseness()

#applyting TOPSIS for the rcloseness
rcs = np.array ([Tm.rCloseness, Ts.rCloseness])
rcs = rcs.T

weightsFinal = np.array ([0.6, 0.4]) # (mean weight, std weigth)
costBenFinal = np.array([0, 0]) #Benefit criteria

Tf = TOPSIS (rcs, weightsFinal, costBenFinal)
Tf.introWeights()
Tf.getIdealSolutions()
Tf.distanceToIdeal()
Tf.relativeCloseness()


print Tf.rCloseness

Alternatives = np.array (['Alg1','Alg2','Alg3','Alg4'])
Tf.plotRankBar(Alternatives)

















