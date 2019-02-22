# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:00:50 2019

@author: Brenden
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as mp
import seaborn as sns; sns.set(color_codes=True)
from scipy import interpolate
from scipy import signal
import math as ma
import os as OS
from statsmodels.tsa.stattools import acf
import scipy.optimize

filename = 'PSI_TSERIES.csv' #input("Input file (include extension but not apostraphes) = ")

funct = np.random.normal(0,1,1000) #Put whatever the signal is here
funct2 = np.random.normal(0,1,1000)


class PSD_Class:
    def __init__(self, func):
        self.func = func

    def corr(self,func):
    	#U_uni1=np.mean(U_uni1)
    	Corr=[]
    	for i in range(len(self.func)):
    		sum1=0
    		U1= self.func[0:(len(self.func)-i)]
    		U2= self.func[(i):len(self.func)]
    		prod=U1*U2
    		sum1=(np.mean(prod))/np.std(self.func)**2
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
    
    
    	 CorrU=self.corr(U_uni)
    	 CorrV=self.corr(V_uni)
    	 #CorrZ=corr(Lag,Z_uni)
    	 CorrU=CorrU/np.max(CorrU)
    	 CorrV=CorrV/np.max(CorrV)
    	 #CorrZ=CorrZ#/np.max(CorrZ)
    	 mp.plot(CorrV[0:1150],label='DNS')
    	 mp.plot(CorrU[0:1150],label='LES')
    	 mp.legend(loc='best')
    	 #mp.plot(CorrZ,label='CorrZ')
#    	 mp.legend(loc='best')
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
    	 y=2.0/N * np.abs(yf[:N//2])
    	 y1=2.0/N*np.abs(yf1[:N//2])
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
    	
    	 FF1=2.0/N*FF1[:N//2]
    	
    	 FF=FF*FF*0.5
    	 FF=2.0/N*FF[:N//2]#np.abs(FF[:N/2])
    	 #print FF
    	 #FF=signal.savgol_filter(FF,139,8)
    	 mp.loglog(xf,FF,label='Velocity')
    	 mp.loglog(xf,FF1,label='Temperature')
    	 C_k=FF[np.size(FF)//2]/(xf[np.size(FF)//2].astype(float))**(-5.0/3.0)
    	 C_k=C_k*0.2
    	 mp.loglog(xf[20:len(xf)-20],C_k*(xf[20:len(xf)-20])**(-5.0/3.0),'--',linewidth=4,color='black')
    	 mp.ylim([10**-8,10**1])
    	 mp.legend(loc='best')
    	 mp.savefig('DNS_LES_2.png',dpi=300,format='png')
        
    	 mp.show()
    #    OS.remove('PSD.pyc')
    
    
    def Taylor(self, dt, file):
        mp.style.use('ggplot')
        mp.rcParams['axes.linewidth'] = 2
        #mp.rcParams['text.usetex'] = True
        mp.rcParams['text.latex.unicode'] = True
        mp.rcParams['font.family'] = 'serif'
        mp.rcParams['axes.linewidth'] = 4
        mp.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
        mp.rcParams.update({'legend.fontsize': 15,
                  'legend.handlelength': 2})
        mp.tick_params(axis='both', which='major', labelsize=12)
        df=pd.read_csv(str(file))
        P=df.values
        A=P[:,np.random.randint(0,len(df.columns))]
        A=pd.Series(A)
        A=A.diff(periods=4)
        A=A.dropna()
        A=A.values
        mp.plot(A)
        mp.show()
        
        dt=float(dt)
        ACF=acf(A,nlags=99)
        ACF=np.array(ACF)
        tau=np.arange(0,100)*dt
        mp.plot(tau,ACF)
        mp.show()
        print(ACF[10])
        osculating = lambda lamb: (ACF[1]-(1-tau[1]**2/lamb))**2
        
        initial=0.1
        lamb = scipy.optimize.fmin(osculating, initial)
        print ("This is the value ",lamb)
        
        mp.plot(tau,ACF[0:len(tau)],color='red',linewidth=4)
        mp.plot(tau,1-tau**2/lamb,'--',color='green',linewidth=4)
        mp.xlabel(r'\textbf{$\tau$(sec)}',fontsize=20)
        mp.ylabel(r'{Autocorrelation}',fontsize=20)
        mp.axvline(x=lamb**0.5,color='black',linestyle='--',linewidth=4)
        mp.axhline(y=0,color='black')
        mp.xlim(0,4)
        mp.annotate(r'\textbf{Osculating-parabola}',fontsize=20, xy=(lamb**0.5, 0),xytext=(1, 0.5),arrowprops=dict(facecolor='black', shrink=0.05))
        mp.annotate('', xy=(0, 0), xycoords='data',
                     xytext=(lamb**0.5, 0), textcoords='data',
                     arrowprops=dict(facecolor='red',lw=4,arrowstyle="<->"))
        mp.annotate(r'\textbf{$\lambda_{\tau}$}', xy=(0.2, 0.05), xycoords='data',
                     fontsize=15.0,textcoords='data',ha='center')
        mp.ylim(-0.3,1.1)
        mp.tight_layout()
        #mp.savefig('Taylor.png',dpi=300,format='png')
        mp.show()






dt = 0.1 #input("dt for Taylor Function = ")
y = PSD_Class(funct)
y.PSD(funct,funct2,np.arange(0,len(funct),1))
mp.plot(y.corr(funct))
y.Taylor(dt,filename)