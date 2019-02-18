# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 12:25:20 2019

@author: bwfeldt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as mp
import seaborn as sns; sns.set(color_codes=True)
from scipy import interpolate
from scipy import signal
import math as ma
import os as OS

class PSD_Class:
    def __init__(self, func):
        self.func = func

    def corr(self, Lag, Vec):
    	Lag = len(self.func)
    	U_uni1= self.func
    	#U_uni1=np.mean(U_uni1)
    	Corr=[]
    	for i in range(len(Lag)):
    		sum1=0
    		U1=U_uni1[0:(len(Vec)-i)]
    		U2=U_uni1[(i):len(Vec)]
    		prod=U1*U2
    		sum1=(np.mean(prod))/np.std(U_uni1)**2
    		sum1=sum1
    		Corr.append(sum1)
    	return np.asarray(Corr)
    
    def PSD(self,U1,V1,Time1):##V1,Z1,Time1):
    	U=U1
    	U=U#-np.mean(U)
    	V=V1
    	V=V#-np.mean(V)
    	print ("This is the shape", np.shape(V))
    	V_uni=V
    	#Z=Z1
    	Time=Time1
    	minT=np.min(Time)
    	maxT=np.max(Time)
    	dt=(maxT-minT)/(len(Time))
    	TimeN=np.arange(minT,maxT,dt)
    	#f1 = interpolate.interp1d(Time, U)
    	#f2= interpolate.interp1d(Time, V)
    	#f3= interpolate.interp1d(Time, Z)
    	#U_uni=f1(TimeN)
    	#mp.plot(Time,U,color='red')
    	#mp.plot(Time,V,color='black')
    	#mp.show()
    	
    	#V_uni=f2(TimeN)
    	#Z_uni=f3(TimeN)
    	N=len(Time)
    	#Autocorrelation
    	Lag=np.arange(0,N)
    	print ('Computing Autocorrelation...')
    	#print Lag
    	U_uni=U
    
    
    	CorrU=corr(Lag,U_uni)
    	CorrV=corr(Lag,V_uni)
    	#CorrZ=corr(Lag,Z_uni)
    	CorrU=CorrU/np.max(CorrU)
    	CorrV=CorrV/np.max(CorrV)
    	#CorrZ=CorrZ#/np.max(CorrZ)
    	mp.plot(CorrU[0:1150],label='LES')
    	mp.plot(CorrV[0:1150],label='DNS')
    	#mp.plot(CorrZ,label='CorrZ')
    	mp.legend(loc='best')
    	mp.xlabel('Lag')
    	mp.ylabel('Autocorrelation')
    	mp.savefig('Autocorr.png',dpi=300,format='png')
    	mp.show()
    	print("Computing energy spectrum...")
    	yf = np.fft.fft(CorrU)#+np.fft.fft(CorrV)+np.fft.fft(CorrZ)
    	yf1= np.fft.fft(CorrV)
    	#yf=signal.savgol_filter(yf,5,4)
    
    	xf = np.linspace(0.0, 1.0/(2.0*dt), N/2)
    	yf=0.5*yf
    	yf1=0.5*yf1
    	fig, ax = mp.subplots()
    	y=2.0/N * np.abs(yf[:N/2])
    	y1=2.0/N*np.abs(yf1[:N/2])
    	#y1=signal.savgol_filter(y1,25,2)
    	#ax.loglog(xf,y,label='LES-Autocorrelation')
    	#ax.loglog(xf,y1,label='DNS-Autocorrelation')
    	#mp.savefig('Autocorr.png',format='png',dpi=300)
    	#mp.show()
    	#FFT and convolution integral
    	print ("Plotting spectrum via multiplication in fourier domain...")
    	FF=(np.fft.fft(U))#+(np.fft.fft(V-np.mean(V)))+(np.fft.fft(Z-np.mean(Z)))
    	FF=np.abs(FF)
    	
    	FF1=(np.fft.fft(V))
    	FF1=np.abs(FF1)
    	FF1=FF1*FF1*0.5
    	
    	FF1=2.0/N*FF1[:N/2]
    	
    	FF=FF*FF*0.5
    	FF=2.0/N*FF[:N/2]#np.abs(FF[:N/2])
    	#print FF
    	#FF=signal.savgol_filter(FF,139,8)
    	mp.loglog(xf,FF,label='Velocity')
    	mp.loglog(xf,FF1,label='Temperature')
    	C_k=FF[np.size(FF)/2]/(xf[np.size(FF)/2].astype(float))**(-5.0/3.0)
    	C_k=C_k*0.2
    	mp.loglog(xf[20:len(xf)-20],C_k*(xf[20:len(xf)-20])**(-5.0/3.0),'--',linewidth=4,color='black')
    	mp.ylim([10**-8,10**1])
    	mp.legend(loc='best')
    	mp.savefig('DNS_LES_2.png',dpi=300,format='png')
        
    	mp.show()
#    OS.remove('PSD.pyc')

nois = np.random.normal(0,1,1000)
