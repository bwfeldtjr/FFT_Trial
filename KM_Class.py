# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 10:43:05 2019

@author: bwfgo
"""

from ggplot import *
#Next 3 are needed for pyplot plotting
#import seaborn as sns; #sns.set(color_codes=True)
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as mp
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import pandas as pd
from sklearn.linear_model import LinearRegression
import ggplotter as ggp
from scipy.interpolate import Rbf
import numpy as np
from sklearn.neighbors.kde import KernelDensity
import math
import scipy as ss


class KM_Class:
    def exponential_fit(self,x, a, b, c):
        return a*np.exp(-b*x) + c

    def corr(self,Lag,Vec):
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

    def KM(self,C,delta,dt,list2=None):# KM Analysis
    	std1=np.std(C)


#    	max1=np.max(C)
#    	min1=np.min(C)
    	#list1=[str(i) for i in range(1,21)]
    	Itemy=[]
    	Itemx=[]
    	Itemx1=[]
    	Itemy1=[]
    	prob=[]
#    	print (len(C))
    	if list2 is None:
            list2=np.arange(0,len(C),int(len(C)/200))#[i for i in range(0,len(C),20)]# int(len(C)/80))]# Divide in 50 segements
    	#print list2
#    	print (np.shape(list2))
    	#print list2
#    	KO=0
#    
#    	Mean12=[]
    	for idx1 in list2:
    
    		for index, item in enumerate(C):
    
    			if abs(item-C[idx1])<=0.1*std1: # Keep it big to have some data points#0.1
    
    				if index<len(C)-delta:
    					Itemx.append(item)
    					Itemy.append(C[index+delta])
    					Itemx1.append(item)
    					Itemy1.append(C[index+delta])
    
    		Itemx1=np.array(Itemx1)
    		Itemy1=np.array(Itemy1)
    
    		hist, bin_edges= np.histogram(Itemy1,bins=len(Itemy1),density=True)

#    		mp.plot(bin_edges[:-1],hist,'o')

    		prob.append(hist)
    		prob.append(Itemx1)
    		prob.append(Itemy1)
    		prob.append(bin_edges)
    
    
    
    		Itemx1=[]
    		Itemy1=[]
    	Itemx=np.array(Itemx)
    	Itemy=np.array(Itemy)

    	if delta==1:
#    	    ax1=mp.subplots()
#    	    ax1.set_ylim(-3,3)
#    	    mp.tight_layout()
#    	    g = sns.jointplot(Itemx/std1, Itemy/std1, kind='kde',data=None,xlim=(-2,2),ylim=(-2,2),n_levels=10)#.plot_joint(sns.kdeplot, zorder=20, n_levels=10)#sns.jointplot(Itemx, Itemy, kind="kde",n_levels=400)
##    	    g.set_axis_ylim(-3,3)
#    	    g.plot_joint(mp.scatter,color='red', s=200, linewidth=0.01, marker='+')
#    	    g.set_axis_labels(r'$v/ \sigma$',r'$v^{\prime}/ \sigma$',fontsize=20)
#    	    mp.tight_layout()
#    	    #mp.savefig('Joint1.png',dpi=300,format='png',transparent=True)
#    	    mp.show()
            
            
#    	    ggplot the density
    	    print(ggp.plotter(Itemx/std1,Itemy/std1)+geom_point(size = 200, shape = '+')+labs(title = 'Density'))
            
            
    	K=[]
    	K1=[]
    	K2=[]
    	K3=[]
    	K4=[]
    	for i in range(0,len(prob),4):
    		K.append(np.mean(prob[i+1]))
    		#print "This is",len(K),len(prob[i+1])
    		K1.append(0.5*np.sum((prob[i+2]-prob[i+1])**2*prob[i]*np.diff(prob[i+3]))/(dt*delta))
    		K2.append(np.sum((prob[i+2]-prob[i+1])**1*prob[i]*np.diff(prob[i+3]))/(dt*delta))
    		K3.append((1.0/6.0)*np.sum((prob[i+2]-prob[i+1])**3*prob[i]*np.diff(prob[i+3]))/(dt*delta))
    		K4.append((1.0/24.0)*np.sum((prob[i+2]-prob[i+1])**4*prob[i]*np.diff(prob[i+3]))/(dt*delta))

    	return(K,K1,K2,K4)

    def Call(self,C,dt, regrnum=None, list2= None):
       Datax=[]
       DataD=[]
       DataD1 = []
       Tdiff=dt*np.arange(0,37)
       #       print (Tdiff)
       for i in range(1,42):
        #           print (i)
           delta=i
           X,D,D1,D4= self.KM(C,delta,dt, list2)
           Datax.append(X)
           DataD.append(D)
           DataD1.append(D1)
       DataD=np.asarray(DataD)
       Datax=np.asarray(Datax)
           
        #       print( np.shape(DataD))
       D_new=[]
       
       #Set up an arbitrary DataFrame and ggplot to add to
       asdf = pd.DataFrame({'x':np.arange(0,5,1)})
       hji = ggplot(aes(),data = asdf)
       
       for j in range(0,len(DataD[0,:])):
           Diffu=DataD[0:37,j]
           Tdiffn=np.linspace(0,37,100)*dt

#           mp.plot(Tdiff,Diffu,'*')

           fgh = pd.DataFrame({'x' : Tdiff,'y' : Diffu})
           hji += geom_point(aes(x='x',y='y'), data = fgh,shape='*', size=50,color='blue')
            
        		#mp.show()
        #           print( np.shape(Tdiff), np.shape(Diffu))
           poly = Rbf(Tdiff, Diffu)#interpolate.splrep(Tdiff,Diffu)
        	#Diffu=interpolate.splev(Tdiffn, poly, der=0)
           Diffu=poly(Tdiffn)
           x=np.asarray(Tdiffn[5:])#30
           y=np.asarray(Diffu[5:])
        	#Tdiffn=Tdiff#np.linspace(0,41,100)*dt
        	#poly = interpolate.splrep(Tdiff,Diffu)#agrange(Tdiff, Diffu)
        		#Diffu=poly(Tdiffn)#interpolate.splev(Tdiffn, poly, der=0)
           regr =  make_pipeline(PolynomialFeatures(int(regrnum)), LinearRegression())#linear_model.LinearRegression()
#           mp.plot(Tdiffn,Diffu,'o')
           
           xyz = pd.DataFrame({'x':Tdiffn,'y':Diffu})
           hji += geom_point(aes(x='x',y='y'), data = xyz, shape ='o', color ='green')
        		#mp.show()

        		#x=Tdiffn[1:]
        		#y=Diffu[1:]
        		#print x,y
        
        	# Train the model using the training sets
           regr.fit(x.reshape(-1,1),y.ravel())#y.reshape(-1,1))
        		#print len(x),len(y)
        		#f = interpolate.interp1d(x, y,fill_value = "extrapolate")
           xnew=np.asarray(Tdiffn[0:])
        		#ynew=regr(xnew)
           y_pred=regr.predict(xnew.reshape(-1,1))
           y_pred=y_pred.ravel()
           D_new.append(y_pred[0])
        		#mp.plot(xnew,ynew,'-',x,y,'o')
        		#mp.scatter(x, y)#  color='black')
#           mp.plot(xnew, y_pred,'--')#, color='blue', linewidth=3)
            
           jkl = pd.DataFrame({'x':xnew,'y':y_pred})
           hji += geom_line(aes(x='x',y='y'), data = jkl, linetype='dashed',color='red')
            
        #    		np.savetxt('REGR.txt',np.asarray(TEST))
        #    		np.savetxt('REGR1.txt',np.asarray(TEST2))
#       mp.show()
        
       print(hji+xlim(low = -.01)+ylim(low = -.001))
       
       return (Datax[0],D_new,DataD1)

    def Everything_else(self,filename,dt,regrnum=3,list2=None):
        Tseries=pd.read_csv(str(filename))#pd.read_csv('Data1.csv')
        #Tseries2=pd.read_csv('PSI_TSERIES_2.csv')
        #Tseries2=Tseries2.values
        #print (np.shape(Tseries))
        RAW=Tseries.values
        RAW=RAW[:,50]
        Tseries=Tseries.values        
        Tseries=Tseries[:,50]#Tseries[:,1]
       
        #np.savetxt('Original.csv',Tseries)
        
        Tseries=pd.Series(Tseries)
        Tseries=Tseries.diff(periods=5)
        Tseries=Tseries.dropna()
        Tseries=Tseries.values

# The Pyplot version of the autocorrelation
#        mp.plot(acf(Tseries,nlags=30))
#        mp.plot(acf(RAW,nlags=30),color='red')
#        mp.title('Auto-Correlation')
#        mp.legend(('Tseries','RAW'))
#        mp.xlabel('Lag')
#        mp.show()
#        #Tseries=Tseries[0:2200]
#        #print (np.shape(Tseries))

#The GGPlot version of the autocorrelation           
#        print(ggp.plotter(np.arange(0,31,1),acf(Tseries,nlags = 30),acf(RAW,nlags=30),ln='Yes')+labs(x='Lag',y=' ',title='Auto-correlation'))

#The Pyplot version of Tseries
#        mp.figure(figsize=(20,5))
#        mp.plot(Tseries)
#        #mp.plot(np.arange(0,120-4/10.0,0.1),Tseries[0:1200],color='black',linewidth=4)
#        mp.title('T Series')
#        mp.xlabel(r'Time',fontsize=20)
#        mp.ylabel(r'Velocity',fontsize=20)
#        
#        mp.show()

#The GGPlot version of Tseries
        
        print(ggp.plotter(np.arange(0,len(Tseries)),Tseries,ln='Yes')+labs(x='Time',y='Velocity',title = 'Tseries'))
        
        Diff=0.01
        dt=dt
#        sample=int(T/dt)
        Diffusion=[]
        X_spa=[]
        D1 = []
        C=Tseries#[:,128]#-np.mean(Tseries[:,128])#[128,:]-np.mean(Tseries[128,:])
        for i in range(0,1):
            X,Diff,D= self.Call(C,dt,int(regrnum),list2)
            X_spa.append(X)
            Diffusion.append(Diff)

# The Pyplot version of X_spa and Diffusion        
#        for i in range(0,1):
#        #		print (np.shape(X_spa[i]))
#        		mp.plot(X_spa[i],Diffusion[i],'o',label='legend')
#        	#mp.ylim([1.25,2])
#        mp.legend(loc='best')
#        #mp.savefig('Diffusion.png',dpi=300,format='png')
#        mp.show()

        

        Diffusion = {'col1': np.asarray(X_spa[0]).ravel(), 'col2': np.asarray(Diffusion[0]).ravel()}
        df=pd.DataFrame(Diffusion)
        df.to_csv('Diffusion_Raw.csv',index=False, header=False)
        
        Drift = {'col1': np.asarray(X_spa[0]).ravel(), 'col2': np.asarray(D[0]).ravel()}
        df=pd.DataFrame(Drift)
        df.to_csv('Drift_Raw.csv',index=False, header=False)

#The GGPlot version of X_spa and Diffusion
#        
#        print(ggp.plotter(np.array(X_spa)[0,:],np.array(Diffusion)[0,:],pt='Yes')+xlim(low=np.min(X_spa)-.01)+ylim(low=0)+labs(x='X_Spa',y='Diffusion'))

    def Reg(self):
        #mp.style.use('ggplot')
#        mp.rcParams['axes.linewidth'] = 2
#        mp.rcParams['text.usetex'] = True
#        mp.rcParams['text.latex.unicode'] = True
#        mp.rcParams['font.family'] = 'serif'
#        mp.rcParams['axes.linewidth'] = 4
#        mp.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
#        mp.rcParams.update({'legend.fontsize': 15,
#                  'legend.handlelength': 2})
#        mp.tick_params(axis='both', which='major', labelsize=12)
        A=pd.read_csv('Diffusion_Raw.csv')
        A=A.values
        xs, ys = zip(*sorted(zip(list(A[:,0]), list(A[:,1]))))
        regr = make_pipeline(PolynomialFeatures(2), LinearRegression())
        regr.fit(np.asarray(xs).reshape(-1,1), np.asarray(ys).reshape(-1,1))
        y_pred=regr.predict(np.asarray(xs).reshape(-1,1))
        mp.plot(np.asarray(xs),y_pred,color='red',linewidth=4,label=r'\textbf{Fitted}')
        np.savetxt('lm_diffusion.txt',[2.14029 ,-0.0156312 ,0.0107945])
        
        #mp.show()
        mp.plot(np.asarray(xs),np.asarray(ys),'o',color='#00FF00',label=r'\textbf{Computed}')
        mp.xlabel(r'\textbf{Velocity-Fluctuations}',fontsize=20)
        mp.ylabel(r'\textbf{Diffusion-coefficient}',fontsize=20)
        mp.tight_layout()
        mp.legend(loc='best')
#        mp.savefig('Diff.png',dpi=300,format='png')
        mp.show()
        
        B=pd.read_csv('Drift_Raw.csv')
        B=B.values
        xs1, ys1 = zip(*sorted(zip(list(B[:,0]), list(B[:,1]))))
        regr = make_pipeline(PolynomialFeatures(4), LinearRegression())
        regr.fit(np.asarray(xs1).reshape(-1,1), np.asarray(ys1).reshape(-1,1))
        y_pred=regr.predict(np.asarray(xs1).reshape(-1,1))
        np.savetxt('lm_drift.txt',[-19.5888, -1.67313, -1.26912, 0.00323181])
        
        mp.plot(np.asarray(xs1),y_pred,color='red',linewidth=4,label=r'\textbf{Fitted}')
        mp.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        #mp.show()
        mp.plot(np.asarray(xs1),np.asarray(ys1),'*',color='#00FF00',label=r'\textbf{Computed}')
        mp.xlabel(r'\textbf{Velocity-Fluctuations}',fontsize=20)
        mp.ylabel(r'\textbf{Drift-coefficient}',fontsize=20)
        mp.tight_layout()
        mp.legend(loc='best')
#        mp.savefig('Drift.png',dpi=300,format='png')
        mp.show()
        
        print( np.shape(A))
        X={'col1':np.asarray(xs),'col2':np.asarray(ys)}
        df=pd.DataFrame(X)
        #print df
        df.to_csv('Diffusion_s.csv',index=False, header=False)
        
        X={'col1':np.asarray(xs1),'col2':np.asarray(ys1)}
        df=pd.DataFrame(X)
        #print df
        df.to_csv('Drift_s.csv',index=False, header=False)
        #A=np.reshape(A,(60,2))
        #mp.plot(A[:,0],A[:,1],'o')
        #mp.show()
#        subprocess.call('./Regression.R')
        A=open('lm_diffusion.txt')
        A=A.readlines()
        B=open('lm_drift.txt')
        B=B.readlines()
        print (np.shape(A))
        def intercept(A):
            Text=[]
            for lines in A:
                K=lines.split()
                #print K
                #print K[0]
                if K:
                    print (K)
                    Text.append(K)
                    #print K[0]
                    #if K[0]=='(Intercept)':
                     #   print K[1]
                    #if K[0]=='poly(X,', '2)1':
                     #   print K[1]
            return(Text)
        Text=intercept(A)
        print (Text[0])
        Text1=intercept(B)
        Intercept1=Text1[3][0]
        C11=Text1[2][0]
        C12=Text1[1][0]
        C13=Text1[0][0]
        #print np.shape(Text1),Text1
        Intercept= Text[2][0]
        C1=Text[1][0]
        C2=Text[0][0]
        #C3=Text[10][2]
        print (Intercept, C1, C2)
        Diffusion=np.asarray([Intercept,C1,C2]).ravel()
        df=pd.DataFrame(Diffusion)
        df.to_csv('Diffusion_para.csv')#,index=False, header=False)
        #Intercept1=Text1[7][1]
        #C11=Text1[8][2]
        #C21=Text1[9][2]
        print (Intercept1, C11, C12, C13)
        Drift=np.asarray([Intercept1,C11,C12,C13]).ravel()
        df=pd.DataFrame(Drift)
        df.to_csv('Drift_para.csv')#,index=False, header=False)


    def Lange(self):
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
        	    if math.isnan(P1):
        	    	    break
        	    if np.isneginf(P1):
        	    	    break
        	    if np.isposinf(P1):
        	    	    break
        	    DIFF.append(P1)
        	    P2=(A+B*X_i**1+C*X_i**2+D*X_i**3)
        	    if math.isnan(P2):
        	    	    break
        	    if np.isneginf(P2):
        	    	    break
        	    if np.isposinf(P2):
        	    	    break
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
            
           
            Diffusion=np.sum(0.1*self.corr(np.arange(0,len(C1)),C1))
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
        	    K1=K1.reshape(-1,1)
        	    K=K.reshape(-1,1)
        
        	    kde = KernelDensity(bandwidth=0.2, kernel='gaussian')
        	    kde.fit(K[0:np.size(K)])
        	    
        	    kde1= KernelDensity(bandwidth=0.2, kernel='gaussian')
        	    kde1.fit(K1[0:np.size(K1)])
        
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
#        U1=U1[:,:]
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
        for i in range(len(U)):
            K3, K4 = zip(*sorted(zip(list(K[i:]), list(K2[i:]))))
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