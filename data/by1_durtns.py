import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.append("..")

t = np.loadtxt('db5_durtns.tsv')
t = t[t > 60.0]
t = t - 60.0
x = np.loadtxt('db5_reboots_durtns.tsv')

from Weibull import *
from Lomax import *

w = Weibull()
w.train_org = w.t = t
w.train_inorg = w.x = x
w.newtonRh()

l = Lomax()
l.train_org = l.t = t
l.train_inorg = l.x = x
l.newtonRh()


def plt_organic_data(data,tail,lmb,k,w_k,w_lmb):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.suptitle('Lomax: lmb:' + "{0:.2f}".format(round(lmb,5)) + ' k:' + "{0:.2f}".format(round(k,5)) + ' Weibull: k:'+ "{0:.2f}".format(round(w_k,5)) + ' lmb:' + "{0:.2f}".format(round(w_lmb,5)))
    [vals, xs, bars] = plt.hist([data,tail],normed=True,stacked=True)
    xs = [(x+xs[i-1])/2 for i,x in enumerate(xs)][1:]
    pdfs = [l.pdf(i,k,lmb) for i in xs]
    plt.plot(xs,pdfs,color='r',label='Lomax fit')
    w_pdfs = [w.pdf(i,w_k,w_lmb) for i in xs]
    plt.plot(xs,w_pdfs,color='y',label='Weibull fit')
    plt.legend()
    ax.set_xlabel("Sample size: " + str(len(data)) + " Generated:" + str(len(tail)))
    plt.show()



