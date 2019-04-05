# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:56:32 2019

@author: Brenden
"""

from ggplot import *
import numpy as np
import PSD_Class as PSD
#import matplotlib.pyplot as mp
import KM_Class as KM
#from mpl_toolkits import mplot3d


filename = 'PSI_TSERIES.csv' #input("Input file (include extension) = ")
dt = 0.1 #input("dt for Taylor Function = ")
regrnum = 3 #input("regrnum for KM_Class")
list2 = np.arange(0,100)


dec1 = 'y' #input("Run PSD_Class (y/n) ")
dec2 = 'y' #input("Run KM_Class (y/n) ")

x = KM.KM_Class()
y = PSD.PSD_Class()

if dec1 == 'y':
    y.Taylor(float(dt),filename)

if dec2 == 'y':
    x.Everything_else(filename,float(dt),int(regrnum), list2)
