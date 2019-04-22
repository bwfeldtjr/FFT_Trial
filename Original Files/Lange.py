# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 10:43:06 2019

@author: bwfgo
"""

import subprocess
import numpy as np
import matplotlib.pyplot as mp
import pandas as pd
import scipy as ss
from sklearn.neighbors.kde import KernelDensity
from statsmodels.tsa.stattools import acf
#mp.style.use('ggplot')
#mp.rcParams['axes.linewidth'] = 2
#mp.rcParams['text.usetex'] = True
#mp.rcParams['text.latex.unicode'] = True
#mp.rcParams['font.family'] = 'serif'
#mp.rcParams['axes.linewidth'] = 4
#mp.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
#mp.rcParams.update({'legend.fontsize': 30,
#          'legend.handlelength': 2})
#mp.tick_params(axis='both', which='major', labelsize=12)

def corr(Lag,Vec):
	Lag=Lag
	U_uni1=Vec
	#U_uni1=np.mean(U_uni1)
	Corr=[]
	for i in Lag:
		sum1=0
		U1=U_uni1[0:(len(Vec)-i)]
		U2=U_uni1[(i):len(Vec)]
		prod=U1*U2
		sum1=(np.mean(prod))/np.std(U_uni1)**2
		sum1=sum1
		Corr.append(sum1)
	return np.asarray(Corr)

def spectrum(T,dt1):
	Tseries_1=T
	FF=(np.fft.fft(Tseries_1))
	FF=FF**2
	N=len(FF)
	dt=dt1
	xf = 2*np.pi*np.linspace(0.0, 1.0/(2.0*dt), int(N/2.0))
	FF=2.0/N*np.abs(FF[:int(N/2.0)])
	#mp.loglog(xf,FF)
	#mp.show()
	return(xf,FF)
#Original=pd.read_csv('Original.csv')
def fun():
    df=pd.read_csv('Diffusion_para.csv')
    df2=pd.read_csv('Drift_para.csv')
    #MEAN=pd.read_csv('smooth.csv')
    #MEAN=MEAN.values
    RAW=pd.read_csv('PSI_TSERIES.csv')
    RAW=RAW.values
    
    RAW=RAW[:,50]
    R0=RAW[0]
    R1=RAW
    RAW=pd.Series(RAW)
    RAW=RAW.diff(periods=1)
    RAW=RAW.dropna()
    RAW=RAW.diff(periods=2)
    RAW=RAW.dropna()
    RAW=RAW.values
    #pcf=acf(RAW,nlags=1000)
    #pcft=np.sum(0.5*np.diff(pcf))*-1
    #print (pcft)
    #mp.show()
    #np.random.seed(30)
    #print np.shape(MEAN[:,128])
    Diffu=df.values
    Drifu=df2.values
    #print Diffu[:,1]
    #print Drifu[:,1]
    A=Drifu[0,1]
    B=Drifu[1,1]
    C=Drifu[2,1]
    D=Drifu[3,1]
    #print A, B, C , D
    mul=1.0

    E=Diffu[0,1]
    F=Diffu[1,1]
    G=Diffu[2,1]
    #subprocess.call('/home/abhinav/PSI_Data/liquid_metal_ux/RAW_LEARNED_ROM/Gen.R')
    #dt=0.1*5
    #print(X)
    #X
    #sample<- 100/0.01
    X=[]

    RAW1=RAW#RAW[0:len(MEAN[:,128]),128]-MEAN[:,128]
    X_i=RAW1[0]
    #np.random.seed(1)
    #print X_i
    T=np.float(len(RAW1))/10.0#np.float(len(MEAN[:,128]))/10.0
    dt=119.5/float(len(RAW1))
    print ("This is the sampling",dt)
    sample=1000#int(T/dt)
    C1=[]
    time=[]
    t=0
    DIFF=[]
    DRIF=[]
    TEMPERA=[]
    ACF=[]
    Temp=0
    XT=1#X_i
    #print A,B,C,D,E,F,G
    for i in range(0,len(RAW)-5):
	    t=t+dt
	    time.append(t)
	    P1=np.sqrt(2*(E+F*X_i**1+G*X_i**2))*np.random.normal(0,1)*(dt*1)**0.5
	    DIFF.append(P1)
	    P2=(A+B*X_i**1+C*X_i**2+D*X_i**3)
	    DRIF.append(P2)
	    #if i==0:
	        #print P1,P2
	    X_i=X_i+P2*(dt*1)+P1
	    
	    #fluc=X_i/XT
	    #print fluc
	    #Temp=Temp+XT*dt+np.sqrt((dt*2*(1.38*10**-23)*274/fluc)+10**-7)*np.random.normal(0,1)
	    #XT=X_i
	    #TEMPERA.append(Temp)
	    #if i==0:
	    #	print("Added")
	    #	C1.append(X_i+R0)
	    #	Temp=X_i+R0
	    #	X_i=Temp
	    C1.append(X_i)
	    #Temp=X_i+Temp
    C1=np.asarray(C1)
    #C1=pd.Series(C1)
    #C1=C1.diff()
    #C1=C1.dropna()
    #C1=C1.values
    
    time=np.asarray(time)
    ACF.append(acf(np.array(C1),nlags=50))

    #sp1,sp2=spectrum(C1,dt)
    #sp3,sp4=spectrum(RAW[0:len(C1)],dt)
    #mp.loglog(sp1,sp2,label='Velocity')
    #mp.loglog(sp3,sp4,label='Temperature')
    #mp.legend(loc=3)
    #mp.loglog(sp1[10:600],10**(2.0)*sp1[10:600]**(-5/3.0),'--',color='black',linewidth=8)
    #mp.text(10**0,10**-0.5,r'\textbf{E$_{11}$$\propto$ f$^\frac{-5}{3}$}',fontsize=20)
    #mp.legend(loc='best')
    #mp.ylabel(r'\textbf{E$_{11}$}',fontsize=20)
    #mp.xlabel(r'\textbf{Frequency}',fontsize=20)
    #mp.tight_layout()
    #mp.grid(True,which="both",ls="-")
    #mp.savefig('spec1.png',dpi=300,format='png')
    #mp.show()
    #mp.plot(corr(np.arange(0,len(C1)),C1))
    #mp.plot(corr(np.arange(0,len(RAW)),RAW))
    
   
    Diffusion=np.sum(0.1*corr(np.arange(0,len(C1)),C1))
    #mp.xlim(0,30)
    #mp.show()
    #UV=pd.read_csv('/home/abhinav/Downloads/Data_Share/Velocity/ux/scale_20/PSI_TSERIES.csv')
    #UV=UV.values
    #UV=UV[:,10]
    mp.plot(acf(RAW),label=r'\textbf{Data}')
    #mp.plot(acf(UV),color='green')
    mp.plot(acf(np.array(C1)),color='red',label=r'\textbf{Model}')
    #mp.ylabel(r'\textbf{Autocorrelation}',fontsize=15)
    #mp.xlabel(r'\textbf{Lag}',fontsize=15)
    #mp.legend(loc='best')
    #mp.savefig('Au1.png',dpi=300,format='png')

    mp.show()
    #mp.figure(figsize=(30,5))

    #mp.plot(D1)
    #mp.show()
    #mp.plot(time,np.array(TEMPERA)+0.1,label=r'\textbf{Regenerated-Temperature}')
    print (len(C1))

    mp.plot(C1,color='black',label=r'\textbf{Regenerated-Velocity}')#+MEAN[0:len(C1),128],color='black',label='Regenerated')
    #mp.show()
    mp.plot(RAW[5:])
    #np.savetxt('MI%s.csv'%i,C1)
    ##mp.plot(Original,color='blue',label=r'\textbf{RAW-nodif}')
    #mp.plot(time,RAW[0:len(C1)],color='red',label=r'\textbf{DNS}')#,128],color='red',label='RAW')#,128])
    #mp.xlabel(r'\textbf{Time}',fontsize=30)
    #mp.ylabel(r'\textbf{Velocity}',fontsize=30)
    mp.legend(loc='best')
    #mp.tick_params(axis='both', which='major', labelsize=30)
    #mp.tight_layout()
    #mp.savefig('Timeseres1.png',dpi=300,format='png',transparent=True)
    mp.show()
    #mp.hist(C1,label='re')
    #mp.legend(loc='best')
    #mp.show()
    #mp.hist(Original,label='Or')
    #mp.legend(loc='best')
    #mp.show()
    #mp.hist(RAW,label='Ra')
    #mp.show()
    #mp.legend(loc='best')
    #mp.show()
    Density={'col1': (C1).ravel(), 'col2': RAW1[0:len(C1)].ravel()}
    df=pd.DataFrame(Density)
    df.to_csv('Density_e.csv')
    xs, ys = zip(*sorted(zip(list(C1), list(DIFF))))
    #subprocess.call('/home/abhinav/PSI_Data/liquid_metal_ux/RAW_LEARNED_ROM/Density.R')
    tau=[1]
    for i in range(0,len(tau)):


	    K=pd.DataFrame(C1)
	    #K=K.diff(periods=tau[i])
	    #K=K.dropna(axis=0)
	    K1=pd.DataFrame(RAW)
	    mp.hist(C1)
	    mp.hist(RAW)
	    mp.show()
	    #K1=K1.diff(periods=tau[i])
	    #K1=K1.dropna(axis=0)
	    #mean=K.mean(axis=0)
	    #mean=mean.values
	    #mean=mean.values/(dt*tau[i])
	    #Mean.append(np.array(mean))

	    K=K.values
	    #K=K-mean
	    K1=K1.values
#	    K1=K1.reshape(-1,1)
	    K=K.reshape(-1,1)
	    print(np.size(K))
	    kde = KernelDensity(bandwidth=0.2, kernel='gaussian')

#	    kde.fit(K[:])
	    
	    kde1= KernelDensity(bandwidth=0.2, kernel='gaussian')
#	    kde1.fit(K1[0:np.size(K1)])

	    logprob = kde.score_samples(K[:])
	    logprob1 = kde1.score_samples(K1[:])
	    ACF=np.asarray(ACF)
	    mp.plot(K,np.exp(logprob),'o')
	    mp.plot(K1,np.exp(logprob1),'*')
	    mp.show()
	    #ACF=np.mean(ACF)
	    #mp.plot(ACF)
	    #mp.plot(acf(RAW,nlags=50))
	    #mp.show()
	    return (K,K1,logprob,logprob1,acf(C1,nlags=50),R1)
U=[]
U1=[]
ACF1=[]
for i in range(1,10):
    print ("This is ",i)
    K,K1,logprob,logprob1,acf1,RAW2=fun()
    ACF1.append(acf1)
    U1.append(K)
    U.append(logprob)
ACF1=np.asarray(ACF1)
print( np.shape(ACF1))
ACF1=np.mean(ACF1,0)
mp.plot(ACF1,'o',color='red',label=r'\textbf{Model}')
mp.plot(acf(RAW2,nlags=50),color='blue',label=r'\textbf{Data}')
#mp.plot(acf(np.array(C1),nlags=50),label=r'\textbf{Model}')
#mp.ylabel(r'\textbf{Autocorrelation}',fontsize=15)
#mp.xlabel(r'\textbf{Lag}',fontsize=15)
#mp.legend(loc='best')
#mp.savefig('Au.png',dpi=300,format='png')
mp.show()
U1=np.asarray(U1)
U1=U1[:,:,0]
#print np.shape(U1)
#U1=np.mean(U1,0)
U=np.asarray(U)
#print np.shape(U)
#U=np.mean(U,0)
print (np.shape(U),np.shape(U1))
K=U1#.ravel()
K2=U#.ravel()
K_mean=[]
K_meanp=[]
KL=[]
I=[]
for i in range(len(U[:,0])):
    K3, K4 = zip(*sorted(zip(list(K[i,:]), list(K2[i,:]))))
    K3=np.asarray(K3)
    K_mean.append(K3)
    
    K4=np.asarray(K4)
    #print "This is the shape of np.exp", np.shape(np.exp(logprob1[0:len(logprob1)-1,None]))
    print (len(np.exp(logprob1[0:len(logprob1),None]).ravel()),len(np.mean(np.asarray(K_meanp),0).ravel()))
    #qk=np.mean(np.asarray(K_meanp),0).ravel()
    #pk=np.exp(logprob1[0:len(logprob1),None]).ravel()
    #print ("This is len",len(qk),len(pk))
    #logprob1=logprob1[0:len(logprob1)-1,None]
     
    K_meanp.append(np.exp(K4))
    qk=np.mean(np.asarray(K_meanp),0).ravel()
    pk=np.exp(logprob1[0:len(logprob1),None]).ravel()
    print ("This is len",len(qk),len(pk))
    pk=pk[0:len(qk)]
    KL.append(ss.stats.entropy(pk,qk))
    I.append(i)
    if i==1:
        mp.plot(K3/np.std(K3),np.exp(K4),color='lawngreen',alpha=0.1,label=r'\textbf{Random-generated}')
    mp.plot(K3/np.std(K3),np.exp(K4),color='lawngreen',alpha=0.1)#,label=r'\textbf{Regenerated}')
mp.plot(K1/np.std(K1),np.exp(logprob1),'*',color='red',label=r'\textbf{Raw-data}')#,label=str(tau[i]))
mp.ylabel(r'\textbf{Probability Density}',fontsize=20)
mp.xlabel(r'\textbf{Velocity/$\sigma$}',fontsize=20)
#mp.title(r'\textbf{Probability}')
K_mean=np.asarray(K_mean)
K_mean=np.mean(K_mean,0)
K_meanp=np.asarray(K_meanp)
K_meanp=np.mean(K_meanp,0)
#print np.shape(K_meanp)
DATA=[]
DATA2=[]
DATA.append(K_mean/np.std(K_mean))
DATA.append(K_meanp)
DATA2.append(np.asarray(K1/np.std(K1)).ravel())
print (np.shape(np.asarray(K1/np.std(K1)).ravel()),np.shape(np.exp(logprob1)))
DATA2.append(np.exp(logprob1).ravel())
print( DATA2)
print( "This is the shape", np.shape(np.asarray(DATA2)))
np.savetxt('Data8.txt',np.asarray(DATA))
np.savetxt('Raw8.txt',np.asarray(DATA2))
mp.plot(K_mean/np.std(K_mean),K_meanp,'+',color='black',label=r'\textbf{Mean}')#s=20,facecolors='none', edgecolors='black',label=r'\textbf{Regenerated-Mean}')#,'+',color='black',label=r'\textbf{Regenerated-Mean}')
#mp.xlim([-4,4])
mp.legend(loc='best')
mp.tight_layout()
mp.savefig('Diff4.png',dpi=300,format='png')



mp.show()    
print ("This is the target",ss.stats.entropy((np.exp(logprob1[0:len(logprob1)-1,None])).ravel()))
maxKL=np.max(np.asarray(KL[200:]))
minKL=np.min(np.asarray(KL[200:]))
fil1=np.ones(len(KL[100:]))*maxKL
fil2=np.ones(len(KL[100:]))*minKL
fil3=np.arange(0,len(KL[100:]))+100
mp.fill_between(fil3,fil1,fil2,color='lawngreen')
mp.plot(np.asarray(I),np.asarray(KL),'+',color='black',markersize=2)
mp.ylabel(r'\textbf{KL-divergence}',fontsize=20)
mp.xlabel(r'\textbf{Number of trajectories}',fontsize=20)
mp.tight_layout()
mp.savefig('KL.png',dpi=300,format='png')
mp.show()

    #K[i+1]=A+B*X[i]+C*X[i]^2+D*X[i]^3
    #K=sqrt((E+F*X[i]^1+G*X[i]^2)*dt)*rnorm(1)
    #print(K)
    #X[i+1]=X[i]+(A+B*X[i]^1+C*X[i]^2+D*X[i]^3)*dt+K#+Tseries1[i]