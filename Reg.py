# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 10:43:07 2019

@author: bwfgo
"""
#from ggplot import *
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

import scipy as ss

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
i = regr.get_params()
np.savetxt('lm_diffusion.txt',[2.14029 ,-0.0156312 ,0.0107945])

y_pred=regr.predict(np.asarray(xs).reshape(-1,1))
mp.plot(np.asarray(xs),y_pred,color='red',linewidth=4,label=r'\textbf{Fitted}')
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
regr = make_pipeline(PolynomialFeatures(3), LinearRegression())
regr.fit(np.asarray(xs1).reshape(-1,1), np.asarray(ys1).reshape(-1,1))
j = regr.get_params()
np.savetxt('lm_drift.txt',[-19.5888, -1.67313, -1.26912, 0.00323181])

y_pred=regr.predict(np.asarray(xs1).reshape(-1,1))
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
Intercept1=float(Text1[3][0])
C11= float(Text1[2][0])
C12=float(Text1[1][0])
C13=float(Text1[0][0])
#print np.shape(Text1),Text1
Intercept= float(Text[2][0])
C1=float(Text[1][0])
C2=float(Text[0][0])
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