# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 12:59:50 2019

@author: bwfeldt
"""

import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0,4001,1)
x = t
sin = np.sin(t*np.pi/180)

ff_t = np.fft.fft(sin)

plt.figure(1)
plt.plot(sin)

plt.figure(2)
plt.plot(ff_t)