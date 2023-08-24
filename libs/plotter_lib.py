import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from scipy import stats
from scipy import signal
from scipy.stats import linregress
import sys

#import seaborn as sb

import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'CMU Serif, Times New Roman'
matplotlib.rc('text', usetex=True)


rgbs={
'lime':"#00ff00",
'fr1':'#009999',
'fr2':'#004c99',
'fr3':'#000099',
'ca1':'#ff0000',
'ca2':'#ff8000',
'ca3':'#ff007f',
}

#import matplotlib
#matplotlib.rcParams['font.family'] = 'serif'
#matplotlib.rcParams['font.serif'] = 'CMU Serif'

class Plotter:
	
	def __init__(self , W=4, H=3):
		
		self.font_axis = 16
		self.font_ticks= 12
		self.font_ticks_inset= 20
		self.font_axis_inset= 25
		
		self.leg_size= 12
		self.W=W
		self.H=H 
		
		self.fig, self.ax = plt.subplots(figsize=(self.W,self.H) )
		self.ax.tick_params(axis="x", labelsize=self.font_ticks)
		self.ax.tick_params(axis="y", labelsize=self.font_ticks)

		self.labx = ''
		self.laby = ''

	def label(self, where='', lab='' ):
		
		if where=='x':
			self.ax.set_xlabel(lab, fontsize=self.font_axis, fontname='CMU Serif')
			
			self.labx = lab
		
		if where=='y':
			self.ax.set_ylabel(lab, fontsize=self.font_axis, fontname='CMU Serif')
			
			self.laby = lab
	
	
	def show(self, save='', loc=1, dpi=300, inset=0, box=False):
		
		if inset:
			
			self.ax.tick_params(axis="x", labelsize=self.font_ticks_inset, direction="in", pad=7,width=2)
			self.ax.tick_params(axis="y", labelsize=self.font_ticks_inset, direction="in", pad=0.1,width=2)
			
			
			self.ax.set_xlabel(self.labx, fontsize=self.font_axis_inset, fontname='CMU Serif')
			self.ax.set_ylabel(self.laby, fontsize=self.font_axis_inset, fontname='CMU Serif')
		
	
		plt.legend(prop={'size': self.leg_size}, loc=loc, fancybox=box, 
			frameon=box, handlelength=self.leg_size*0.1)
	
		
		plt.tight_layout()
		
		if save!='':
			
			plt.savefig(save, dpi=dpi, transparent=True)
		
		plt.show()
	
	
	
	def animar_curvas(self, data, xlims=[None,None], ylims=[None,None], scale='lin'):
		
		
		self.ax.set_xlim(xlims[0],xlims[1])
		self.ax.set_ylim(ylims[0],ylims[1])
		
		pi, = self.ax.plot([], [], marker='o', linewidth=0.5)
		
		ps = [pi]
		
		
		def init():
			ps[0].set_data([], [])
			return ps

		def animate(i):
			d = np.array(data[i])
			
			x,y = d[:,0], (1.0*d[:,1])/sum(d[:,1])
			
			if scale=='log':
				x,y = np.log10(d[:,0]), np.log10( (1.0*d[:,1])/sum(d[:,1]) )
			else:
				x,y = d[:,0], (1.0*d[:,1])/sum(d[:,1])
			
			ps[0].set_data(x,y)
			
			if not i%1:
				print ('t = %d'%i, len(x), len(y))

			return ps


		anim = FuncAnimation(self.fig, animate, init_func=init, frames=len(data), interval=80, blit=True)	
			
		
	
	
	
	
	
	def save(self, save='x.svg', loc=1, dpi=300, inset=0, box=False, transparent=True):
		
		plt.legend(prop={'size': self.leg_size}, loc=loc, fancybox=box, 
			frameon=box, handlelength=self.leg_size*0.1)
		
		plt.tight_layout()
		plt.savefig(save, dpi=dpi, transparent=transparent)



	def logscale(self, eje='',p1=[1e-4, 1e4], p2= [1e4, 1e-4]):
		
		
		if eje=='':
			self.ax.set_xscale('log')
			self.ax.set_yscale('log')	
		
		if eje=='x':
			self.ax.set_xscale('log')
		
		if eje=='y':
			self.ax.set_yscale('log')
		
		
		x0 = p1[0]
		x1 = p2[0]
		
		y0 = p2[1]
		y1 = p1[1]
	
		self.ax.set_xlim(x0, x1)	
		self.ax.set_ylim(y0, y1)



def get_dist(dist, pars, norm=1, zeros=0):

	minimo = pars[0]
	maximo = pars[1]
	
	ancho_bin = pars[2]
	
	x=minimo
	n=0

	bines = [x]

	while x < maximo:
		x += ancho_bin
		bines.append(x)
		n+=1
	
	
	y0, x0 = np.histogram(dist, bins= bines)

	xp = []

	for i in range(0,len(x0)-1):
		
		b0 = x0[i]
		b1 = x0[i+1]
		xp.append( b0+ 0.5*(b1-b0) )
		

	xp= np.array(xp)
	
	if norm:
		yp= 1.0*y0/sum(y0)
	else:
		yp= 1.0*y0
	
	
	if zeros:
		return xp, yp
	else:
		return xp[yp>0], yp[yp>0]


def get_dist_log_bins(dist, pars, n0=0, norm=1,zeros=0):
	
	
	# defino bines
	tmin = pars[0]
	tmax = pars[1]
	a= pars[2]
	
	t=tmin
	n=n0

	bines = [ ]

	while t < tmax:
		t = np.power(a, n)
		bines.append(t)
		n+=1


	#print bines

	y0, x0 = np.histogram(dist, bins= np.array(bines) )

	xp ,yp = [],[]

	for i in range(0,len(x0)-1):
		
		b0 = x0[i]
		b1 = x0[i+1]
		xp.append( b0+ 0.5*(b1-b0) )
		yp.append(1.0*y0[i]/(b1-b0))
		

	xp= np.array(xp)
	yp= np.array(yp)
	
	if norm:
		yp= 1.0*yp/sum(yp)
	else:
		None

	if zeros:
		return xp, yp
	else:
		return xp[yp>0], yp[yp>0]


def get_dist_log_bins2(dist, pars, n0=0):
	
	
	# defino bines
	tmin = pars[0]
	tmax = pars[1]
	a= pars[2]
	
	t=tmin
	n=n0

	bines = [ ]

	while t < tmax:
		t = np.power(a, n)
		bines.append(t)
		n+=1


	#print bines
	N = len(bines)-1
	y0, x0 = np.zeros(N),bines
	
	for i in range(len(dist)):
		
		flag=1
		nb=0
		
		while flag and nb<(len(bines)-1):
			
			a,b = bines[nb], bines[nb+1]
			
			if dist[i]>=a and dist[i]<b:
				
				flag=0
				
				y0[nb]+=1
				
			nb+=1
			
			
	
	xp ,yp = [],[]

	for i in range(0,len(x0)-1):
		
		b0 = x0[i]
		b1 = x0[i+1]
		xp.append( b0+ 0.5*(b1-b0) )
		yp.append(1.0*y0[i])#/(b1-b0))
		
		
	xp= np.array(xp)
	yp= np.array(yp)
	
	yp= 1.0*yp/sum(yp)
	
	print (sum(yp))

	return xp[yp>0], yp[yp>0]






def fit_lineal(i0=0, i1=1, x=[], y=[] ):
	
	'''
	# Para fiteos de leyes de potencia recordar:
	
	k2,w2 = np.log(k), np.log(w)
	a,b= np.log(0.5), np.log(300)

	r,index = fit_lineal(i0=a, i1=b, x=k2, y=w2 )
	print (r)
	
	fl = lambda u: np.exp(r.intercept)*(u**r.slope)
	
	# ----------------------------------
	'''
	
	
	from scipy.stats import linregress

	condition = np.logical_and( np.greater(x,i0), np.less(x, i1 ) )
	index = np.where(condition)
	
	print (index)

	r = linregress(x[index],y[index])

	return r, index


def gen_log_space(limit, n):
    result = [1]
    if n>1:  # just a check to avoid ZeroDivisionError
        ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    while len(result)<n:
        next_value = result[-1]*ratio
        if next_value - result[-1] >= 1:
            # safe zone. next_value will be a different integer
            result.append(next_value)
        else:
            # problem! same integer. we need to find next_value by artificially incrementing previous value
            result.append(result[-1]+1)
            # recalculate the ratio so that the remaining values will scale correctly
            ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
    return np.array(list(map(lambda x: round(x)-1, result)), dtype=np.uint64)



# para el analisis de series *************

def	get_dfa(x, s1=1, s2=5):
	
	# paso 1 centrar acumular ----------
	xmean = np.mean(x)
	acu, y =0, []

	for i in range(len(x)):
		acu+= (x[i]-(xmean))
		y.append(acu)

	y = np.array(y)
	t= np.arange(0,len(y),1)
	
	
	# paso 2 chunk --------------------
	chunks = []
	T= t[-1]
	n=T

	while n>10:
		chunks.append(n)
		n= n/2.0

	# ----------------------------------

	S, FS = [],[]

	for chunk in chunks:
		
		tref = 0			
		std_sum =0
		std_con =0
		
		while (tref+chunk)<=T:
			condition = np.logical_and( np.greater(t,tref), np.less(t,(tref+chunk) ) )
			index = np.where(condition)
							
			x2 = t[index]
			y2 = y[index]
			
			if len(x2)>2 and len(y2)>2:

				r = stats.linregress(x2,y2)
				yt = x2*r.slope + r.intercept
				
				for i in range(len(x2)):
					std_sum+= (y2[i]-yt[i])**2
					std_con+=1
				

			tref = tref+chunk
		
		if std_con:
			
			fs = np.sqrt( 1.0*std_sum/std_con)
			
			S.append( np.log10(chunk) )
			FS.append( np.log10(fs) )


	S, FS = np.array(S), np.array(FS)
	
	condition = np.logical_and( np.greater(S,s1), np.less(S, s2) ) 
	index = np.where(condition)
	
	
	
	
	r = stats.linregress(S[index],FS[index])
	
	return r.slope,S,FS



def get_rs(p):

	lags = range(2,100)

	variancetau = []; tau = []

	for lag in lags: 

		#  Write the different lags into a vector to compute a set of tau or lags
		tau.append(lag)

		# Compute the log returns on all days, then compute the variance on the difference in log returns
		# call this pp or the price difference
		pp = np.subtract(p[lag:], p[:-lag])
		variancetau.append(np.var(pp))

	# we now have a set of tau or lags and a corresponding set of variances.
	#print tau
	#print variancetau

	# plot the log of those variance against the log of tau and get the slope
	m = np.polyfit(np.log10(tau),np.log10(variancetau),1)

	hurst = m[0] / 2.

	return hurst



def autocorr(x):
	
	T= len(x)
	tau=0
		
	x1,y1=[],[]
	
	while T/(tau+1) > 2:
		
		c1,c2,c3 =[],[],[]
		
		for t in range(T-tau):
			
			c1.append(x[t+tau]*x[t])
			c2.append(x[t+tau])
			c3.append(x[t])
		
		corr = (np.mean(c1) - (np.mean(c2)*np.mean(c3)))/np.std(c2)/np.std(c3)
		
		x1.append(tau)
		y1.append(corr)
		
		tau+=1
	
	
		
	x1= np.array(x1)
	y1= np.array(y1)

	
	return x1,y1


def crosscorr(xi,xj):
	
	T= len(xi)
	tau=0
		
	x1,y1=[],[]
	
	while T/(tau+1) > 2:
		
		c1,c2,c3 =[],[],[]
		
		for t in range(T-tau):
			
			c1.append(xi[t+tau]*xj[t])
			c2.append(xi[t+tau])
			c3.append(xj[t])
		
		corr = (np.mean(c1) - (np.mean(c2)*np.mean(c3)))/np.std(c2)/np.std(c3)
		
		x1.append(tau)
		y1.append(corr)
		
		tau+=1
	
	
		
	x1= np.array(x1)
	y1= np.array(y1)

	
	return x1,y1
	
def get_psd(d, fs=1, nperseg=1024*2):
	
	N = len(d)
		
	f, psd = signal.welch(d, fs=fs, nperseg=nperseg, scaling= 'spectrum')
	x = f[psd>0]
	y = psd[psd>0]
		
	return x,y
	

# interevent times

def get_inter_event_times(x,y, umbral):
	
	tiempo = x
	e = y
		
	tinter=[]
	
	for i in range(len(tiempo)-1):
		
		if (e[i]<=umbral and e[i+1]>umbral) or (e[i]>=umbral and e[i+1]<umbral):
			
			tinter.append( tiempo[i] + 0.5*(tiempo[i+1]-tiempo[i])  )
	
	dt = [ tinter[i+1]-tinter[i] for i in range(len(tinter)-1)]

	return np.array(dt)


def get_inter_event_times_2(x,y, umbral):
	
	tiempo = x
	e = y
		
	tinter=[]
	
	for i in range(len(tiempo)-1):
		
		if (e[i]<=umbral and e[i+1]>umbral) or (e[i]>=umbral and e[i+1]<umbral):
			
			tinter.append( tiempo[i] )#+ 0.5*(tiempo[i+1]-tiempo[i])  )
	
	tp = [ tinter[i+1]-tinter[i] for i in range(0,len(tinter)-1,2)]
	
	tr = [ tinter[i+2]-tinter[i] for i in range(0,len(tinter)-2,2)]

	

	tf = np.array(tinter)[::2]

	return np.array(tp),np.array(tr), tf



# print en misma linea
def printr(st):
	sys.stdout.write("\r%s"%st)
	sys.stdout.flush()



def get_ellipse(x1, y1):
	
	x1=np.array(x1)
	y1=np.array(y1)
	
	from scipy.stats import linregress
	r = linregress(x1,y1)

	fl = lambda x: r.slope*x + r.intercept
	
	x1r = x1
	y1r = y1-fl(x1)
	
	x1m, x1s = np.mean(x1r), np.std(x1r)
	y1m, y1s = np.mean(y1r), np.std(y1r)
	
	t = np.linspace(0, 2*np.pi, 100)
	x1e = x1s*np.cos(t)+x1m
	y1e = y1s*np.sin(t)+y1m
	
	
	return x1e, (y1e + fl(x1e))


def get_ellipse2(x1, y1):
	
	x1=np.array(x1)
	y1=np.array(y1)
	
	from scipy.stats import linregress
	r = linregress(x1,y1)

	fl = lambda x: r.slope*x + r.intercept
	
	x1r = x1
	y1r = y1-fl(x1)
	
	x1m, x1s = np.mean(x1r), np.std(x1r)
	y1m, y1s = np.mean(y1r), np.std(y1r)
	
	t = np.linspace(0, 2*np.pi, 100)
	x1e = x1s*np.cos(t)+x1m
	y1e = y1s*np.sin(t)+y1m
	
	
	return x1e, (y1e + fl(x1e)), x1s, y1s


def crear_carpeta(directory):
	import os
	if not os.path.exists(directory):
		os.makedirs(directory)





class Potencial:
	
	
	def __init__(self , z1, z2, dl=0.1):
		
		z1min, z1max = min(z1), max(z1)
		z2min, z2max = min(z2), max(z2)
		
		xref, yref = z1min-dl*0.5, z2min-dl*0.5
		
		data={}
		
		
		for i in range(len(z1)):
			
			xref, yref = z1min-dl*0.5, z2min-dl*0.5

			#---------------------------------------
			lx,flag=0,1
			while (xref< (z1max+dl*0.5)) and flag:
				
				lx+=1
		
				if z1[i]>xref and z1[i]<=(xref+dl):
					flag=0
					x=xref
				
				xref+=dl
				
				
			ly,flag=0,1	
			while (yref< (z2max+dl*0.5)) and flag:
				
				ly+=1
				if z2[i]>yref and z2[i]<=(yref+dl):
					flag=0
					y=yref
				
				
				yref+=dl
					
			#---------------------------------------

			try:
				data[(lx,ly)]['count']+=1
			except:
				data[(lx,ly)]= {'count':1, 'xy':(x,y)}
			

		#print (data)

		d =[]
		for ij, pij in sorted(data.items(), key=lambda x:x[0], reverse=True):
			d.append( [ij[0], ij[1], data[ij]['xy'][0], data[ij]['xy'][1], data[ij]['count']] )
		
		self.d= np.array(d)
		
		m,n = int(max(self.d[:,0]))+1, int(max(self.d[:,1]))+1

		#print (n,m)

		self.M = np.zeros((n,m))
		for i in range(self.d.shape[0]):
			self.M[ int(self.d[i,1]), int(self.d[i,0]) ]= self.d[i,-1]
			
		
		self.dl = dl
		
		self.potmax= max(self.d[:,4])
		
		

	def get_pxy(self,xi,yi):
		
		
		for i in range(self.d.shape[0]):
			
			xref, yref = self.d[i,2], self.d[i,3]
			
			if xi>xref and yi>yref:
			
				if xi<=xref+self.dl and yi<=yref+self.dl:
					
					return self.d[i,4]/self.potmax
				


		return 0.0







def Dkl(P,Q):
	
	dkl=0
	
	for i in range(len(P)):
		
		if Q[i]>0 and P[i]>0:
			dkl+= P[i]*np.log(P[i]/Q[i])
		

	return dkl
	

def Djs(d1,d2):
	
	n = min(len(d1), len(d2))
	
	P, Q = [],[]
	
	for i in range(n):
		
		if(d1[i,0] == d2[i,0]):
			
			P.append(d1[i,1])
			Q.append(d2[i,1])
	
	
	P = np.array(P)
	Q = np.array(Q)
	
	M = 0.5*(P+Q)
	
	return (100.0*len(P)/n), (0.5*Dkl(P,M) + 0.5*Dkl(Q,M))




def get_mean_std(d):
	
	x=d[:,0]
	y=d[:,1]
	
	N=len(x)
	x_, x2_ =0,0
	
	for i in range(N):
		x_ += x[i]*y[i]
		x2_ += (x[i]**2)*y[i]
	
	mean = x_
	sigma= np.sqrt( x2_ - mean**2 )
	
	return mean,sigma





def animacion_X1X2(X1,X2):
		
	plot = Plotter(W=5, H=5)
	plot.ax.set_xlim(-50,50)
	plot.ax.set_ylim(-50,50)


	ps = []
	for i in range(2):
		pi, = plot.ax.plot([], [], marker='o', linewidth=0, c="C%d"%i)
		ps.append(pi)


	def init():
		
		for i in range(2):
			ps[i].set_data([], [])
		
		return ps




	def animate(i):
		
		x1,x2 = X1[i], X2[i]
		
		
		ps[0].set_data( x1[:,0], x1[:,1] )
		ps[1].set_data( x2[:,0], x2[:,1] )
		
		
		return ps



	anim = FuncAnimation(plot.fig, animate, init_func=init,
								   frames=(X1.shape[0]-1), interval=300, blit=True)	
	plot.show()



def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]




def get_stats(X, info=0):
	
	if len(X)>0:
		s = {"len":len(X), "min":min(X), "max":max(X), "mean":np.mean(X), "std":np.std(X), "var":np.var(X)}
		
		if info:
			print ("\nget_stats info ________________________________________________________________")
			print ("len,min,max :", s["len"],s["min"],s["max"])
			print ("mean,std,var :", s["mean"],s["std"],s["var"])
			print ("_______________________________________________________________________________")
		
		return s
	
	else:
		print ("[get_stats ERROR :Array vacio]")
		return {}



def avg_tex(x):
	return "\left \langle %s  \right\rangle"%x



