# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 14:40:01 2019

@author: Brenden
"""
#from ggplot import *
import seaborn as sns; #sns.set(color_codes=True)
import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.interpolate import Rbf
import matplotlib.pyplot as mp
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import pandas as pd
from sklearn.linear_model import LinearRegression
#import ggplotter as ggp
#sns.set(rc={'figure.figsize':(20,10)})


class KM_Class:
    def exponential_fit(self,x, a, b, c):
        return a*np.exp(-b*x) + c

    def KM(self,C,delta,dt):# KM Analysis
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
    	    g = sns.jointplot(Itemx/std1, Itemy/std1, kind='kde',data=None,xlim=(-2,2),ylim=(-2,2),n_levels=10)#.plot_joint(sns.kdeplot, zorder=20, n_levels=10)#sns.jointplot(Itemx, Itemy, kind="kde",n_levels=400)
#    	    g.set_axis_ylim(-3,3)
    	    g.plot_joint(mp.scatter,color='red', s=200, linewidth=0.01, marker='+')
    	    g.set_axis_labels(r'$v/ \sigma$',r'$v^{\prime}/ \sigma$',fontsize=20)
    	    mp.tight_layout()
    	    #mp.savefig('Joint1.png',dpi=300,format='png',transparent=True)
    	    mp.show()
            
            
#    	    dens = pd.DataFrame({'x':Itemx/std1,'y':Itemy/std1})
#    	    tes10 = ggplot(aes(x='x',y='y'),data = dens) + geom_point(size = 200, shape = '+')
#    	    print(tes10)

#    	    print(ggp.plotter(Itemx/std1,Itemy/std1)+geom_point(size = 200, shape = '+'))
            
            
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

    def Call(self,C,dt):
       Datax=[]
       DataD=[]
       Tdiff=dt*np.arange(0,37)
       #       print (Tdiff)
       for i in range(1,42):
        #           print (i)
           delta=i
           X,D,D1,D4= self.KM(C,delta,dt)
           Datax.append(X)
           DataD.append(D)
       DataD=np.asarray(DataD)
       Datax=np.asarray(Datax)
           
        #       print( np.shape(DataD))
       D_new=[]
#       asdf = pd.DataFrame({'x':np.arange(0,5,1)})
#       hji = ggplot(aes(),data = asdf)
       for j in range(0,len(DataD[0,:])):
           Diffu=DataD[0:37,j]
#           for i in range(0,37):
#        		#print np.shape(DataD[i])
#               Diffu.append(DataD[i][j])
           Tdiffn=np.linspace(0,37,100)*dt
           mp.plot(Tdiff,Diffu,'*')
#           fgh = pd.DataFrame({'x' : Tdiff,'y' : Diffu})
#           hji += geom_point(aes(x='x',y='y'), data = fgh,shape='*', size=50,color='blue')
            
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
           regr =  make_pipeline(PolynomialFeatures(3), LinearRegression())#linear_model.LinearRegression()
           mp.plot(Tdiffn,Diffu,'o')
#           xyz = pd.DataFrame({'x':Tdiffn,'y':Diffu})
#           hji += geom_point(aes(x='x',y='y'), data = xyz, shape ='o', color ='green')
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
           mp.plot(xnew, y_pred,'--')#, color='blue', linewidth=3)
            
#           jkl = pd.DataFrame({'x':xnew,'y':y_pred})
#           hji += geom_line(aes(x='x',y='y'), data = jkl, linetype='dashed',color='red')
            
        #    		np.savetxt('REGR.txt',np.asarray(TEST))
        #    		np.savetxt('REGR1.txt',np.asarray(TEST2))
       mp.show()
        
#       print(hji+xlim(low = -.1)+ylim(low = -.01))
       
       return (Datax[0],D_new)

    def Everything_else(self,filename,dt):
        #from bokeh.layouts import gridplot
        #from bokeh.plotting import figure, show, output_file
        Tseries=pd.read_csv(str(filename))#pd.read_csv('Data1.csv')
        #Tseries2=pd.read_csv('PSI_TSERIES_2.csv')
        #Tseries2=Tseries2.values
        #print (np.shape(Tseries))
        RAW=Tseries.values
        RAW=RAW[:,50]
        Tseries=Tseries.values
        
        #mp.imshow(Tseries)
        #mp.show()
        #mp.plot(Tseries[:,50])
        #mp.show()
        
        Tseries=Tseries[:,50]#Tseries[:,1]
        #np.savetxt('Original.csv',Tseries)
        
        Tseries=pd.Series(Tseries)
        #mp.plot(Tseries2[0:1200],color='red')
        #mp.plot(Tseries[0:1200])
        #mp.show()
        #Tseries_new=Tseries.diff(periods=20)
        #Tseries_new=Tseries_new.dropna()
        Tseries=Tseries.diff(periods=5)
        Tseries=Tseries.dropna()
        #smooth=smooth.values
        Tseries=Tseries.values

# The Pyplot version of the autocorrelation
        mp.plot(acf(Tseries,nlags=30))
        mp.plot(acf(RAW,nlags=30),color='red')
        mp.title('Auto-Correlation')
        mp.legend(('Tseries','RAW'))
        mp.xlabel('Lag')
        mp.show()
#        #Tseries=Tseries[0:2200]
#        #print (np.shape(Tseries))

#The GGPlot version of the autocorrelation        
#        RAWT = pd.DataFrame({'Lag':np.arange(0,31,1),'Tseries':acf(Tseries,nlags=30),'RAW':acf(RAW,nlags=30)})
#        tes5 = ggplot(aes(),data=RAWT) + geom_line(aes(x='Lag',y='Tseries'),data=RAWT,color = 'blue') + geom_line(aes(x='Lag',y='RAW'), data=RAWT, color = 'red') + labs(x='Lag',y=' ',title='Auto-correlation') #+ theme(legend.position = 'right')
#        print(tes5)
        
#        print(ggp.plotter(np.arange(0,31,1),acf(Tseries,nlags = 30),acf(RAW,nlags=30),ln='Yes')+labs(x='Lag',y=' ',title='Auto-correlation'))

#The Pyplot version of Tseries
        mp.figure(figsize=(20,5))
        mp.plot(Tseries)
        #mp.plot(np.arange(0,120-4/10.0,0.1),Tseries[0:1200],color='black',linewidth=4)
        mp.title('T Series')
        mp.xlabel(r'Time',fontsize=20)
        mp.ylabel(r'Velocity',fontsize=20)
        
        mp.show()

#The GGPlot version of Tseries
#        TTS = pd.DataFrame({'Time':np.arange(0,len(Tseries)),'Tseries':Tseries})
#        tes8 = ggplot(aes(x='Time',y='Tseries'), data = TTS) + geom_line() + labs(y = 'Velocity',title = 'T Series')
#        print(tes8)
        
#        print(ggp.plotter(np.arange(0,len(Tseries)),Tseries,ln='Yes')+labs(x='Time',y='Velocity',title = 'Tseries'))
        
        #print (np.shape(Tseries))
        Diff=0.01
        drift=0.001
        X_i=0
        T=100.0
        dt=dt
        sample=int(T/dt)
        C=[]
        time=[]
        t=0
        
        Diffusion=[]
        X_spa=[]
        C=Tseries#[:,128]#-np.mean(Tseries[:,128])#[128,:]-np.mean(Tseries[128,:])
        for i in range(0,1):
            X,Diff= self.Call(C,dt)
            X_spa.append(X)
            Diffusion.append(Diff)
            
# The Pyplot version of C
#        mp.plot(C,'*')
#        mp.legend('Tseries')
#        #mp.plot(Tseries2)
#        mp.show()
        
#The GGPlot version of C        
#        CCC = pd.DataFrame({'x':np.arange(0,len(C),1),'y':C})
#        tes6 = ggplot(aes(x='x',y='y'), data=CCC) + geom_point()
#        print(tes6)
        

# The Pyplot version of X_spa and Diffusion        
        for i in range(0,1):
        #		print (np.shape(X_spa[i]))
        		mp.plot(X_spa[i],Diffusion[i],'o',label='legend')
        	#mp.ylim([1.25,2])
        mp.legend(loc='best')
        #mp.savefig('Diffusion.png',dpi=300,format='png')
        mp.show()
        

#The GGPlot version of X_spa and Diffusion
#        XsD = pd.DataFrame({'X_Spa':np.array(X_spa)[0,:],'Diffusion':np.array(Diffusion)[0,:]})
#        tes7 = ggplot(aes(x='X_Spa',y='Diffusion'), data=XsD) + geom_point() +xlim(low=np.min(X_spa)-.01)+ylim(low=0)
#        print(tes7)
        
#        print(ggp.plotter(np.array(X_spa)[0,:],np.array(Diffusion)[0,:],pt='Yes')+xlim(low=np.min(X_spa)-.01)+ylim(low=0)+labs(x='X_Spa',y='Diffusion'))
        
        #Diffusion = {'col1': np.asarray(X_spa[0]).ravel(), 'col2': np.asarray(Diffusion[0]).ravel()}
        #df=pd.DataFrame(Diffusion)
        
        #K=pd.DataFrame(C)
        #dl=1000
        #Mean=[]
        #tau=np.array([1,10,100,900])
        
        #	K=pd.DataFrame(C)
        #	K=K.diff(periods=tau[i])
        #	K=K.dropna(axis=0)
        #	mean=K.mean(axis=0)
        #	mean=mean.values/(dt*tau[i])
        #	Mean.append(np.array(mean))
        
        #	K=K.values
        
        #	K=K.reshape(-1,1)
        
        #	kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
        #	kde.fit(K[0:np.size(K)/2])
        
        #	logprob = kde.score_samples(K[:])
        #	mp.plot(K/np.std(K),np.exp(logprob),'o',label=str(tau[i]))
        #mp.ylabel(r'$Probability$')
        #mp.xlabel(r'$x$')
        #mp.title(r'$Probability$')
        #mp.legend(loc='best')
        #mp.savefig('Diff4.png',dpi=300,format='png')
        
        
        
        #mp.show()


