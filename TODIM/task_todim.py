# -*- coding: utf-8 -*-
"""
Author: Andre Pacheco
Email: pacheco.comp@gmail.com

An example of how to use the TODIM class.

"""

from TODIM import TODIM
import numpy as np

A = TODIM ('decisionMatrix.txt')
A.getRCloseness(verbose=True)
A.plotBars(['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10'],'Fig1')


# If you don't wanna use the file .txt, you can set the values 
# as lists or numpy arrays
