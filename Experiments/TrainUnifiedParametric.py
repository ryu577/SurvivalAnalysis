import numpy as np
import scipy
import pandas as pd
import datetime
from scipy.special import erf
import sys
sys.path.append('../')
from Weibull import *
from Lomax import *
from LogNormal import *
from LogLogistic import *

features = []
ys = []

def plot_haz(t, x, i):
	l = Lomax()
	w = Weibull()
	ln = Lognormal()
	ll = Loglogistic()
	l.train_org = t
	l.train_inorg = x
	w.train_org = t
	w.train_inorg = x
	ln.train_org = t
	ln.train_inorg = x
	ll.train_org = t
	ll.train_inorg = x
	l.newtonRh()
	w.newtonRh()
	ln.gradient_descent()
	#ll.newtonRh(numIter = 3001)
	ll.gradient_descent(numIter=20001)
	plt = plt_organic_data(t,x,w,l,ln,ll)
	plt.savefig("E:\\git\\SurvivalAnalysis\\Plots\\Recovery\\" + str(i) + ".png")
	plt.close()
	rebootCost = 300.0
	plt = plt_haz_rates(w,l,ln,ll,rebootCost)
	nonParametric = non_parametric2(t,x) * 60.0
	plt.axvline(nonParametric, color = "blue")
	plt.savefig("E:\\git\\SurvivalAnalysis\\Plots\\HazRates2\\" + str(i) + ".png")
	plt.close()
	features.append(np.array([l.k, l.lmb, w.k, w.lmb, w.optimal_threshold(300.0), ll.alp, ll.beta, ln.mu, ln.sigma]))
	ys.append(nonParametric)

def extend(x):
	return np.concatenate((x, np.ones(len(x)/3)*np.mean(x)), axis=0)

'''
Finds the optimal threshold given some data and making no assumption about the distribution.
'''
def non_parametric2(t, x, rebootCost = 300.00):
	recoveryTimes = t
	relSavings = []
	numOfReboots = len(x)
	taus = np.arange(60,600)/60.0
	indx = 0
	neglosses = []
	poslosses = []
	for tau in taus:
		indx = indx + 1
		savings = numOfReboots * (600.0 - tau * 60.0)
		losses = 0
		for i in recoveryTimes:
			if i > tau * 60.0 and i < 600.0:
				losses = losses + (tau * 60.0 + rebootCost - i)
		netSavings = (savings - losses)
		relSavings.append(netSavings)
		if indx%10 == 0:
			print "tau: " + "{0:.2f}".format(tau) + " savings: " + "{0:.2f}".format(savings) + " losses: " + "{0:.2f}".format(losses) + " net: " + "{0:.2f}".format(netSavings)
	#print "Optimal threshold: " + str(taus[np.argmax(relSavings)])
	return taus[np.argmax(relSavings)]


def plots_on_random(samples):
	for i in range(samples):
		t = np.concatenate((np.random.uniform(size=30)*500 , (100+np.random.uniform(size=30)*100)), axis=0)
		x = np.ones(10)*600.0
		plot_haz(t, x, str(i))


