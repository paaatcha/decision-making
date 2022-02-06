# -*- coding: utf-8 -*-
"""
Author: Andre Pacheco
Email: pacheco.comp@gmail.com

An example of how to use the TODIM class.

"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from TODIM import TODIM
import numpy as np

A = TODIM ('decisionMatrix.txt')
A.getRCloseness(verbose=True)
A.plotBars(['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10'])


# If you don't wanna use the file .txt, you can set the values 
# as lists or numpy arrays

# [[0.        ]
#  [0.02486829]
#  [0.39209586]
#  [0.31041422]
#  [0.93245574]
#  [0.3727109 ]
#  [0.74886502]
#  [0.99960448]
#  [1.        ]
#  [0.92957362]]
