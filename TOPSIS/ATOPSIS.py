from TOPSIS import TOPSIS
import numpy as np

#applying TOPSIS to means
Tm = TOPSIS ('means.txt')
Tm.introWeights()
Tm.getIdealSolutions()
Tm.distanceToIdeal()
Tm.relativeCloseness()

#applying TOPSIS to std
Ts = TOPSIS ('std.txt')
Ts.introWeights()
Ts.getIdealSolutions()
Ts.distanceToIdeal()
Ts.relativeCloseness()

#applyting TOPSIS for the rcloseness
rcs = np.array ([Tm.rCloseness, Ts.rCloseness])
rcs = rcs.T

weightsFinal = np.array ([0.7, 0.3]) # (mean weight, std weigth)
costBenFinal = np.array([0, 0]) #Benefit criteria

Tf = TOPSIS (rcs, weightsFinal, costBenFinal)
Tf.introWeights()
Tf.getIdealSolutions()
Tf.distanceToIdeal()
Tf.relativeCloseness()


print Tf.rCloseness


















