import numpy as np
import scipy
import pandas as pd
import datetime
from scipy.special import erf
from Weibull import *
from Lomax import *
from LogNormal import *
from LogLogistic import *

def plt_organic_data(data,tail,w,l,ln,ll):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.suptitle('Lomax: lmb:' + "{0:.2f}".format(round(l.lmb,5)) + ' k:' + "{0:.2f}".format(round(l.k,5)) + ' Weibull: k:'+ "{0:.2f}".format(round(w.k,5)) + ' lmb:' + "{0:.2f}".format(round(w.lmb,5)))
    [vals, xs, bars] = plt.hist([data,tail],normed=True,stacked=True)
    xs = [(x+xs[i-1])/2 for i,x in enumerate(xs)][1:]
    pdfs = [l.pdf(i,l.k,l.lmb) for i in xs]
    plt.plot(xs,pdfs,color='r',label='Lomax fit')
    w_pdfs = [w.pdf(i,w.k,w.lmb) for i in xs]
    plt.plot(xs,w_pdfs,color='y',label='Weibull fit')
    exp_lmb = 1/np.mean(data)
    e_pdfs = [ln.pdf(i,ln.mu,ln.sigma) for i in xs]
    plt.plot(xs,e_pdfs,color='purple',label='Log normal fit')
    ll_pdfs = [ll.pdf(i,ll.alp,ll.beta) for i in xs]
    plt.plot(xs,ll_pdfs,color='orange',label='Log logistic fit')
    plt.legend()
    ax.set_xlabel("Sample size: " + str(len(data)) + " Generated:" + str(len(tail)))
    plt.show()

def plt_haz_rates():
	fig = plt.figure()
	ax = fig.add_subplot(111)
	indx = 0
	cols = ['g','r','purple','orange']
	ln.mu = np.mean(np.log(t))
	ln.sigma = np.var(np.log(t))**.5
	for i in [w,l,ln,ll]:
		xs = np.arange(0.1,9.99,0.1)*60
		ys = i.pdf(xs,i.params[0],i.params[1])/i.survival(xs,i.params[0],i.params[1])
		plt.plot(xs,ys,color=cols[indx])
		indx=indx+1
	plt.axhline(1/rebootCost)
	plt.show()

## Censoring succesively to the optimal threshold and moving it each time.
def succesive_censoring(numIter = 60, Cint = 420):
	thresholds = [] # These will store treatment and control arrays.
	thresholds2 = []
	lens = []
	l1 = Lomax() # Two lomaxes, one with censoring and one without.
	l2 = Lomax()
	l1.k = 0.9 # Set the parameters to values we have seen in the data.
	l1.lmb = 2.5
	samples = l1.samples(l1.k,l1.lmb, 20000) # Generate samples.
	opt_thresh = (Cint * l1.k - 1/l1.lmb)
	thresholds.append(opt_thresh) # Initially, set treatment and control to initial threshold.
	thresholds2.append(opt_thresh)
	for i in range(numIter):
		samples = l1.samples(0.9,2.5, 20000)
		l1.train_org = samples[samples < opt_thresh]
		l1.train_inorg = np.array([opt_thresh] * sum(samples > opt_thresh))
		lens.append(sum(samples > opt_thresh))
		l1.newtonRh() # Train first Lomax after censoring.
		l2.train_org = samples
		l2.train_inorg = np.array([.1,.1])
		l2.newtonRh() # Train second lomax before censoring.
		opt_thresh = (Cint * l1.k - 1/l1.lmb)
		opt_thresh2 = (Cint * l2.k - 1/l2.lmb) # Update the optimal thresholds
		thresholds.append(opt_thresh)
		thresholds2.append(opt_thresh2) # Save to arrays so we can plot.
	return [thresholds, thresholds2, lens]

def plot_succesive_censoring():
	# Simulation to see how succesive censoring affects the optimal thresholds.
	thresholds = succesive_censoring(60, 10)
	plt.plot(thresholds[0])
	plt.plot(thresholds[1])
	plt.show()

if __name__ == '__main__':
	ln = Lognormal()
	t = ln.train_org
	x = ln.train_inorg
	ln.gradient_descent(numIter=201)
	w = Weibull()
	l = Lomax()
	w.train_org = ln.train_org
	w.train_inorg = ln.train_inorg
	l.train_org = ln.train_org
	l.train_inorg = ln.train_inorg
	w.newtonRh()
	l.newtonRh()
	ll = Loglogistic()
	#Takes too long to do gradient descent, so lets just hard code the parameters.
	ll.alp = 256.37343165
	ll.beta = 3.7150498
	ll.params = np.array([256.37343165, 3.7150498])
	
