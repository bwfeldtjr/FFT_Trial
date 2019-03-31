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


filename = 'PSI_TSERIES.csv' #input("Input file (include extension but not apostraphes) = ")
dt = 0.1 #input("dt for Taylor Function = ")


y = PSD.PSD_Class()
y.Taylor(dt,filename)

x = KM.KM_Class()
x.Everything_else(filename,dt)
