import numpy as np
import matplotlib.pyplot as plt

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

t = np.concatenate((np.random.uniform(size=30)*900 , (200+np.random.uniform(size=20)*200)), axis=0)





