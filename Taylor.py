import numpy as np
import matplotlib.pyplot as mp
import scipy.optimize
import pandas as pd
from statsmodels.tsa.stattools import acf
#mp.style.use('ggplot')
mp.rcParams['axes.linewidth'] = 2
mp.rcParams['text.usetex'] = True
mp.rcParams['text.latex.unicode'] = True
mp.rcParams['font.family'] = 'serif'
mp.rcParams['axes.linewidth'] = 4
mp.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
mp.rcParams.update({'legend.fontsize': 15,
          'legend.handlelength': 2})
mp.tick_params(axis='both', which='major', labelsize=12)
df=pd.read_csv('PSI_TSERIES.csv')
P=df.values
A=P[:,50]
A=pd.Series(A)
#A=A.diff(periods=4)
#A=A.dropna()
A=A.values
#mp.plot(A)
#mp.show()
def corr(Lag,Vec):
	Lag=Lag
	U_uni1=Vec
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
print( np.mean(A))
dt=0.1
Taylor=[]
Time=[]
ACF=acf(A,nlags=100)
ACF=np.array(ACF)
tau=np.arange(0,100)*dt
#mp.plot(tau,ACF)
#mp.show()
print(ACF[10])
osculating = lambda lamb: (ACF[1]-(1-tau[1]**2/lamb))**2

initial=0.1
lamb = scipy.optimize.fmin(osculating, initial)
print ("This is the value ",lamb)

mp.plot(tau,ACF[0:len(tau)],color='red',linewidth=4)
mp.plot(tau,1-tau**2/lamb,'--',color='green',linewidth=4)
mp.xlabel(r'\textbf{$\tau$(sec)}',fontsize=20)
mp.ylabel(r'{Autocorrelation}',fontsize=20)
#mp.axvline(x=lamb**0.5,color='black',linestyle='--',linewidth=4)
#mp.axhline(y=0,color='black')
mp.xlim(0,4)
mp.annotate(r'\textbf{Osculating-parabola}',fontsize=20, xy=(lamb**0.5, 0),xytext=(1, 0.5),arrowprops=dict(facecolor='black', shrink=0.05))
mp.annotate('', xy=(0, 0), xycoords='data',
             xytext=(lamb**0.5, 0), textcoords='data',
             arrowprops=dict(facecolor='red',lw=4,arrowstyle="<->"))
mp.annotate(r'\textbf{$\lambda_{\tau}$}', xy=(0.2, 0.05), xycoords='data',
             fontsize=15.0,textcoords='data',ha='center')
mp.ylim(-0.15,1.1)
mp.tight_layout()
#mp.savefig('Taylor.png',dpi=300,format='png')
mp.show()




