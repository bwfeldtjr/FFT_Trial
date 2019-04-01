import seaborn as sns; #sns.set(color_codes=True)
from scipy import stats
from scipy import signal
import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.interpolate import Rbf
from sklearn.svm import SVR
from scipy import interpolate
#from scipy.interpolate import lagrange
import matplotlib.pyplot as mp
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
#mp.style.use('dark_background')
import pandas as pd
from sklearn import  linear_model
from scipy.optimize import curve_fit
from sklearn.neighbors.kde import KernelDensity
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from matplotlib import rc
from scipy.stats import norm, gaussian_kde
from statsmodels.distributions.empirical_distribution import ECDF
#sns.set(rc={'figure.figsize':(20,10)})
#mp.style.use('ggplot')
mp.rcParams['axes.linewidth'] = 20
#mp.rcParams['text.usetex'] = True
mp.rcParams['text.latex.unicode'] = True
mp.rcParams['font.family'] = 'serif'
mp.rcParams['axes.linewidth'] = 8
mp.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
mp.tick_params(axis='both', which='major', labelsize=12)

def exponential_fit(x, a, b, c):
    return a*np.exp(-b*x) + c

def KM(C,delta):# KM Analysis
	std1=np.std(C)

	max1=np.max(C)
	min1=np.min(C)
	#list1=[str(i) for i in range(1,21)]
	Itemy=[]
	Itemx=[]
	Itemx1=[]
	Itemy1=[]
	prob=[]
	print (len(C))
	list2=np.arange(0,len(C),int(len(C)/200))#[i for i in range(0,len(C),20)]# int(len(C)/80))]# Divide in 50 segements
	#print list2
	print (np.shape(list2))
	#print list2
	KO=0

	Mean12=[]
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
		#mp.plot(bin_edges[:-1],hist,'o')
		prob.append(hist)
		prob.append(Itemx1)
		prob.append(Itemy1)
		prob.append(bin_edges)



		Itemx1=[]
		Itemy1=[]
	Itemx=np.array(Itemx)
	Itemy=np.array(Itemy)
	#mp.show()
	#mp.plot(Itemx, Itemy,'o')
	#mp.show()
	if delta==1:
#	    ax1=mp.subplots()
#	    ax1.set_ylim(-3,3)
	    mp.tight_layout()
	    g = sns.jointplot(Itemx/std1, Itemy/std1, kind='kde',data=None,xlim=(-2,2),ylim=(-2,2),n_levels=10)#.plot_joint(sns.kdeplot, zorder=20, n_levels=10)#sns.jointplot(Itemx, Itemy, kind="kde",n_levels=400)
	    g.set_axis_ylim(-3,3)
	    g.plot_joint(mp.scatter,color='red', s=200, linewidth=0.01, marker='+')
	    g.set_axis_labels(r'$v/ \sigma$',r'$v^{\prime}/ \sigma$',fontsize=20)
	    mp.tight_layout()
#	    mp.savefig('Joint1.png',dpi=300,format='png',transparent=True)
	    mp.show()
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
	#print 'Lengths',len(K),len(K1)
	#mp.plot(K1)
	#mp.show()
	#mp.plot(K,K1,'o',color='red',label='Computed')
	#mp.ylabel(r'$Diffusion$')
	#mp.xlabel(r'$x$')
	#mp.plot(C,C**2+1,'*',linewidth=1,color='blue', markersize=2,label='True')
	#mp.legend(loc='best')
	#mp.savefig('Diff.png',dpi=300,format='png')
	#mp.show()
	#mp.plot(K,K2,'o',color='red',label='Computed')
	#mp.ylabel(r'$Drift$')
	#mp.xlabel(r'$x$')
	#mp.plot(C,-C**3+C,'*',linewidth=1,color='blue', markersize=2,label='True')
	#mp.legend(loc='best')
	#mp.savefig('Diff1.png',dpi=300,format='png')
	#mp.show()
	return(K,K1,K2,K4)
TEST=[]
TEST2=[]	
def Call(C):
	Datax=[]
	DataD=[]
	Tdiff=dt*np.arange(0,37)
	print (Tdiff)
	for i in range(1,42):
		print (i)
		#if i==0:
		#	C1=C
		#if i>0:
			#C1=np.diff(C,i)
		delta=i
		X,D,D1,D4=KM(C,delta)
		#mp.show()
		Datax.append(X)
		DataD.append(D)
	DataD=np.asarray(DataD)
	Datax=np.asarray(Datax)
	#print DataD[8]
	#print DataD[8][-1]
	#mp.plot(Datax[0],Datax[1])
	#mp.show()
	#for i in range(0,1):
	#	mp.plot(Datax[i],DataD[i],'o',label=str(i))
	#mp.legend(loc='best')
	#mp.show()
	#mp.plot(Datax[0][0:10],Datax[1][0:10])
	#mp.show()
	#print Datax[0][0], Datax[1][0]
	print( np.shape(DataD))
	D_new=[]
	for j in range(0,len(DataD[0,:])):
		Diffu=[]
		for i in range(0,37):
			#print np.shape(DataD[i])
			Diffu.append(DataD[i][j])
		Tdiffn=np.linspace(0,37,100)*dt
		mp.plot(Tdiff,Diffu,'*')
		#mp.show()
		print( np.shape(Tdiff), np.shape(Diffu))
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
		#mp.show()
		TEST.append(Tdiffn)
		TEST.append(Diffu)
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
		TEST2.append(xnew)
		TEST2.append(y_pred)
		#mp.plot(Tdiff,Diffu)#,label=str(j))
		#mp.xlabel(r'$\Delta \tau$')
		#mp.ylabel(r'\textbf{Drift}')
		#mp.xlim(0,2)
		#mp.savefig('reg.png',dpi=300,format='png')
		#mp.show()
		np.savetxt('REGR.txt',np.asarray(TEST))
		np.savetxt('REGR1.txt',np.asarray(TEST2))
	mp.show()
	return (Datax[0],D_new)



#from bokeh.layouts import gridplot
#from bokeh.plotting import figure, show, output_file
Tseries=pd.read_csv('PSI_TSERIES.csv')#pd.read_csv('Data1.csv')
#Tseries2=pd.read_csv('PSI_TSERIES_2.csv')
#Tseries2=Tseries2.values
print (np.shape(Tseries))
RAW=Tseries
RAW=RAW.values
RAW=RAW[:,50]
Tseries=Tseries.values
#mp.imshow(Tseries)
#mp.show()
mp.plot(Tseries[:,50])
mp.show()
Tseries=Tseries[:,50]#Tseries[:,1]
#np.savetxt('Original.csv',Tseries)
Tseries=pd.Series(Tseries)
#mp.plot(Tseries2[0:1200],color='red')
#mp.plot(Tseries[0:1200])
mp.show()
#Tseries_new=Tseries.diff(periods=20)
#Tseries_new=Tseries_new.dropna()
Tseries=Tseries.diff(periods=5)
#Tseries=Tseries.dropna()
#smooth=Tseries.rolling(10).mean()#.fillna(0)
#smooth=smooth.dropna()
#smooth.to_csv('smooth.csv')
#print np.shape(smooth)
#Tseries=Tseries-smooth
Tseries=Tseries.dropna()

#smooth=smooth.values
Tseries=Tseries.values
mp.plot(acf(Tseries,nlags=30))
mp.plot(acf(RAW,nlags=30),color='red')
mp.show()
#Tseries=Tseries[0:2200]
print (np.shape(Tseries))
mp.figure(figsize=(20,5))
mp.plot(Tseries)
#mp.plot(np.arange(0,120-4/10.0,0.1),Tseries[0:1200],color='black',linewidth=4)
mp.xlabel(r'Time',fontsize=20)
mp.ylabel(r'Velocity',fontsize=20)
#mp.plot(Tseries_new+1,color='blue',linewidth=4)
#mp.xticks([])
#mp.yticks([])
#mp.savefig('Measured.png',dpi=300,format='png',transparent=True)
mp.show()
#mp.plot(Tseries[:,10])
#print np.mean(Tseries[:,10],axis=0)
#mp.plot(smooth[:,10])
#mp.show()
#Tseries1=pd.read_csv('Nearwall_POD.csv')
#Tseries1=Tseries1.values
#Tseries2=pd.read_csv('Awaywall_POD.csv')
#Tseries2=Tseries2.values
print (np.shape(Tseries))
Diff=0.01
drift=0.001
X_i=0
T=100.0
dt=0.1
sample=int(T/dt)
C=[]
time=[]
t=0
#for i in range(0,sample):
#	t=t+dt
#	time.append(t)
#	X_i=X_i+(-X_i**3+X_i)*dt+np.sqrt(2*(X_i**2+1)*dt)*np.random.normal(0,1)
#	C.append(X_i)
#C=np.asarray(C)



	#mp.plot(Datax[0],np.asarray(D_new),'o')
	#mp.show()
Diffusion=[]
X_spa=[]
for i in range(0,1):
	C=Tseries#[:,128]#-np.mean(Tseries[:,128])#[128,:]-np.mean(Tseries[128,:])
	X,Diff=Call(C)
	X_spa.append(X)
	Diffusion.append(Diff)
	mp.plot(C,'*')
	#mp.plot(Tseries2)
	mp.show()

for i in range(0,1):
		print (np.shape(X_spa[i]))
		mp.plot(X_spa[i],Diffusion[i],'o',label=str(i))
	#mp.ylim([1.25,2])
mp.legend(loc='best')
#mp.savefig('Diffusion.png',dpi=300,format='png')
mp.show()
Diffusion = {'col1': np.asarray(X_spa[0]).ravel(), 'col2': np.asarray(Diffusion[0]).ravel()}
df=pd.DataFrame(Diffusion)
#df.to_csv('Diffusion_Raw.csv',index=False, header=False)
#mp.plot(K,K3,'o',color='red',label='Computed')
#mp.ylabel(r'$3^{rd}$ $Moment$')
#mp.xlabel(r'$x$')
#mp.savefig('Diff2.png',dpi=300,format='png')
#mp.show()
#mp.plot(K,K4,'o',color='red',label='Computed')
#mp.ylabel(r'$4^{th}$ $Moment$')
#mp.xlabel(r'$x$')
#mp.savefig('Diff3.png',dpi=300,format='png')
#mp.show()

#TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select"
#p1 = figure(title="Legend Example", tools=TOOLS)
#p1.circle(K,K1,color='orange')
#output_file("legend.html", title="legend.py example")
#show(p1)  # open a browser

K=pd.DataFrame(C)
dl=1000
Mean=[]
tau=np.array([1,10,100,900])
#for i in range(0,len(tau)):


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