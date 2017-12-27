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
    ln_pdfs = [ln.pdf(i,ln.mu,ln.sigma) for i in xs]
    plt.plot(xs,ln_pdfs,color='purple',label='Log normal fit')
    ll_pdfs = [ll.pdf(i,ll.alpha,ll.beta) for i in xs]
    plt.plot(xs,ll_pdfs,color='orange',label='Log logistic fit')
    plt.legend()
    ax.set_xlabel("Sample size: " + str(len(data)) + " Generated:" + str(len(tail)))
    #plt.show()
    return plt

def plt_haz_rates(w,l,ln,ll,rebootCost):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	indx = 0
	cols = ['g','r','purple','orange']
	for i in [w,l,ln,ll]:
		xs = np.arange(0.1,9.99,0.1)*60
		ys = i.pdf(xs,i.params[0],i.params[1])/i.survival(xs,i.params[0],i.params[1])
		plt.plot(xs, ys, color=cols[indx])
		indx=indx+1
	plt.axhline(1/rebootCost, color="pink")
	#plt.show()
	return plt

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
	thresholds.append(opt_thresh)
	thresholds2.append(opt_thresh)
	for i in range(numIter):
		samples = l1.samples(0.9,2.5, 20000)
		l1.train_org = samples[samples < opt_thresh]
		l1.train_inorg = np.array([opt_thresh] * sum(samples > opt_thresh))
		lens.append(sum(samples > opt_thresh))
		l1.newtonRh()
		l2.train_org = samples
		l2.train_inorg = np.array([.1,.1])
		l2.newtonRh()
		opt_thresh = (Cint * l1.k - 1/l1.lmb)
		opt_thresh2 = (Cint * l2.k - 1/l2.lmb)
		thresholds.append(opt_thresh)
		thresholds2.append(opt_thresh2)
	return [thresholds, thresholds2, lens]

def plot_succesive_censoring():
	# Simulation to see how succesive censoring affects the optimal thresholds.
	thresholds = succesive_censoring(60, 10)
	plt.plot(thresholds[0])
	plt.plot(thresholds[1])
	plt.show()

def main_run():
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

def extract_durations(data):
	durtns = np.array(data.ix[:,2])
	states = np.array(data.ix[:,3])
	counts = np.array(data.ix[:,4])
	t = []
	x = []
	for i in xrange(len(counts)):
		for j in range(counts[i]):
			if states[i] == "Ready":
				t.append(durtns[i])
			elif states[i] == "PoweringOn":
				x.append(durtns[i])
	t = np.array(t)
	x = np.array(x)
	###### Comment out next two lines
	t = np.array(durtns[states == "Ready"])
	x = np.array(durtns[states == "PoweringOn"])
	######
	t = t[t > 60] - 60
	x = x[x > 60] - 60
	return [t,x]

def plot_all():
	data = pd.read_csv("F:\data\cosmosTest\\v8_backup\\nst.csv", sep = '\t')
	## Need to read as tab saperated and account for lack of headers.
	l = Lomax()
	w = Weibull()
	ln = Lognormal()
	ll = Loglogistic()
	for i in set(data.ix[:,0]): #Loop through all clusters
	#for i in ["AM2PrdApp02","AM2PrdApp04","AM2PrdApp05","AM2PrdApp06","AM2PrdApp07","AM3PrdApp01","AM3PrdApp02","AM3PrdApp06","AM3PrdApp07","AM4PrdApp10","AM4PrdApp13","AM4PrdApp18","AM4PrdApp20","AM4PrdApp22","AM5PrdApp01","AM5PrdApp02","AM5PrdApp03","AM5PrdApp06","AM5PrdApp08","AM5PrdApp24","AM5PrdApp33","AM5PrdApp40","AM5PrdApp48","AM5PrdApp51","BL2PrdApp04","BL2PrdApp05","BL2PrdApp06","BL2PrdApp10","BL2PrdApp11","BL2PrdApp14","BL2PrdApp15","BL2PrdApp20","BL3PrdApp01","BL3PrdApp02","BL3PrdApp03","BL3PrdApp04","BL3PrdApp06","BL3PrdApp07","BL3PrdApp10","BL3PrdApp11","BL3PrdApp12","BL4PrdApp01","BL4PrdApp10","BL4PrdApp20","BL5PrdApp05","BL5PrdApp06","BL5PrdApp08","BL5PrdApp14","BL5PrdApp17","BL5PrdApp19","BL5PrdApp23","BL5PrdApp25","BL5PrdApp26","BL5PrdApp28","BL5PrdApp29","BL5PrdApp30","BL5PrdApp31","BL5PrdApp35","BL5PrdApp37","BL5PrdApp41","BL6PrdApp03","BL6PrdApp05","BL6PrdApp07","BL6PrdApp13","BL6PrdApp17","BL6PrdApp18","BL6PrdApp19","BL6PrdApp20","BLUPrdApp05","BN3PrdApp02","BN3PrdApp05","BN3PrdApp07","BN4PrdApp02","BN4PrdApp11","BN4PrdApp13","BN6PrdApp12","BN6PrdApp23","BY1PrdApp02","BY1PrdApp10","BY2PrdApp03","BY2PrdApp05","BY2PrdApp06","BY2PrdApp08","BY3PrdApp09","BY3PrdApp15","BY3PrdApp17","BY4PrdApp01","BY4PrdApp03","BY4PrdApp04","BY4PrdApp08","BY4PrdApp09","BY4PrdApp12","BY4PrdApp14","BY4PrdApp15","BY4PrdApp17","BY4PrdApp19","BY4PrdApp20","BY4PrdApp22","BY4PrdApp27","BY4PrdApp29","BY4PrdApp43","BY4PrdApp46","BY4PrdApp50","BY4PrdApp52","BYAPrdApp04","BYAPrdApp09","BYAPrdApp12","BYAPrdApp13","BYAPrdApp18","BYAPrdApp24","BZ6PrdApp03","CBN06PrdApp01","CH1PrdApp02","CH1PrdApp04","CH1PrdApp10","CH1PrdApp11","CH1PrdApp14","CH1PrdApp23","CH3PrdApp04","CH3PrdApp05","CH3PrdApp07","CQ1PrdApp01","CQ1PrdApp02","CQ1PrdApp03","CQ2PrdApp05","CQ2PrdApp06","CW1PrdApp03","CY4PrdApp01","CY4PrdApp03","CY4PrdApp04","CY4PrdApp05","DB3PrdApp04","DB3PrdApp09","DB4PrdApp04","DB4PrdApp07","DB4PrdApp08","DB4PrdApp14","DB5PrdApp03","DB5PrdApp18","DB5PrdApp19","DB5PrdApp20","DB5PrdApp28","DB5PrdApp29","DB6PrdApp01","DB6PrdApp04","DB6PrdApp07","DB6PrdApp13","DB6PrdApp18","DB6PrdApp20","DB6PrdApp22","DM2PrdApp01","DM2PrdApp04","DM2PrdApp08","DM3PrdApp01","DM3PrdApp02","DM3PrdApp04","DM3PrdApp07","DM3PrdApp27","DM3PrdApp29","DM5PrdApp01","DM5PrdApp02","DM5PrdApp05","DM5PrdApp08","DM5PrdApp12","DM5PrdApp16","DM5PrdApp17","DZ5PrdApp01","DZ5PrdApp04","DZ5PrdApp05","DZ5PrdApp08","HK2PrdApp02","HK2PrdApp03","HK2PrdApp10","HK2PrdApp13","HK2PrdApp26","HK2PrdApp27","HKNPrdApp01","HKNPrdApp03","HKNPrdApp04","KW1PrdApp01","LN1PrdApp01","LN1PrdApp03","LO1PrdApp01","MA1PrdApp03","ML1PrdApp01","ML1PrdApp04","ML1PrdApp07","ML1PrdApp09","ML1PrdApp10","MWH01PrdApp10","MWH01PrdApp11","MWH01PrdApp12","MWH01PrdApp13","OS1PrdApp02","OS2PrdApp01","PN1PrdApp01","PS1PrdApp02","SG1PrdApp02","SG1PrdApp05","SG1PrdApp06","SG1PrdApp07","SG1PrdApp09","SG2PrdApp06","SG2PrdApp16","SG2PrdApp17","SG2PrdApp18","SG2PrdApp27","SG2PrdApp29","SG2PrdApp34","SG3PrdApp01","SN1PrdApp01Agg02","SN1PrdApp05","SN2PrdApp05","SN2PrdApp10","SN2PrdApp15","SN3PrdApp01","SN3PrdApp02","SN3PrdApp07","SN3PrdApp10","SN3PrdApp22","SN3PrdApp24","SN3PrdApp26","SN4PrdApp13","SN4PrdApp14","SN4PrdApp18","SN4PrdApp22","SN4PrdApp24","SN4PrdApp26","SY3PrdApp04","SY3PrdApp08","SY3PrdApp09","SY3PrdApp10","SY3PrdApp11","SY3PrdApp13","SY3PrdApp14","SY3PrdApp15","SY3PrdApp16","TY1PrdApp01","TY1PrdApp02","TY1PrdApp07","TY1PrdApp08","TY1PrdApp09","TY1PrdApp12","TY1PrdApp14","YQ1PrdApp03","YQ1PrdApp04","YT1PrdApp07"]:
		data_filtered = data[data.ix[:,0] == i]
		[t,x] = extract_durations(data_filtered)
		if len(t) > 3 and len(x) > 2:
			try:
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
				ll.newtonRh(numIter = 3001)
				plt = plt_organic_data(t,x,w,l,ln,ll)
				plt.savefig("E:\\git\\SurvivalAnalysis\\Plots\\Clusters\\" + i + ".png")
				plt.close()
				rebootCost = 300.0
				plt = plt_haz_rates(w,l,ln,ll,rebootCost)
				nonParametric = non_parametric(w) * 60.0
				plt.axvline(nonParametric, color = "blue")
				plt.savefig("E:\\git\\SurvivalAnalysis\\Plots\\HazRates\\" + i + ".png")
				plt.close()
			except:
				continue

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
	ll.newtonRh(numIter = 3001)
	plt = plt_organic_data(t,x,w,l,ln,ll)
	plt.savefig("E:\\git\\SurvivalAnalysis\\Plots\\Recovery\\" + str(i) + ".png")
	plt.close()
	rebootCost = 300.0
	plt = plt_haz_rates(w,l,ln,ll,rebootCost)
	nonParametric = non_parametric2(t,x) * 60.0
	plt.axvline(nonParametric, color = "blue")
	plt.savefig("E:\\git\\SurvivalAnalysis\\Plots\\HazRates2\\" + str(i) + ".png")
	plt.close()


def extend(x):
	return np.concatenate((x, np.ones(len(x)/3)*np.mean(x)), axis=0)

'''
Finds the optimal threshold given some data and making no assumption about the distribution.
'''
def non_parametric(w):
	recoveryTimes = w.train_org + 60.0 # Add 60.0 because UnhealthyTransitionLowerBound was subtracted.
	rebootCost = 300.00
	relSavings = []
	numOfReboots = len(w.train_inorg)
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


def plots_on_random():
	for i in range(10):
		t = np.concatenate((np.random.uniform(size=30)*500 , (100+np.random.uniform(size=30)*100)), axis=0)
		x = np.ones(10)*600.0
		plot_haz(t, x, str(i))



