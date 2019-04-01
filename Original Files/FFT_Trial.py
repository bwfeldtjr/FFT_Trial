# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:00:49 2019

@author: Brenden
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