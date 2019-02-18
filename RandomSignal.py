# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 13:28:46 2019

@author: bwfeldt
"""

import scipy.fftpack as scf
import numpy as np
import matplotlib.pyplot as plt

def corr(Lag, Vec):
	Lag = Lag
	U_uni1= Vec
	#U_uni1=np.mean(U_uni1)
	Corr=[]
	for i in range(Lag):
		sum1=0
		U1=U_uni1[0:(len(Vec)-i)]
		U2=U_uni1[(i):len(Vec)]
		prod=U1*U2
		sum1=(np.mean(prod))/np.std(U_uni1)**2
		sum1=sum1
		Corr.append(sum1)
	return np.asarray(Corr)
nois = np.random.normal(0,1,1000)
x = corr(1000,nois)

plt.plot(x)

#plt.figure(1)
#plt.plot(nois/max(abs(nois)))
#
#ff = np.fft.fft(nois)
#plt.figure(2)
#plt.plot(ff/max(abs(ff)))
#
#ff2= scf.fft(nois)
#plt.figure(3)
#plt.plot(ff2/max(abs(ff2)))
