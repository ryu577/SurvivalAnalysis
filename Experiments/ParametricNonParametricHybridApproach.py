import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from Weibull import *
from Lomax import *
from LogNormal import *
from LogLogistic import *
from MarkovChains import *
from data_gen import *

## Single parameter cost function for validation.
def cost(tau, l, scale = None, shape = None, rebootCost = 199.997):
  Et = l.expectedT(tau)
  probs = np.array(
    [
      [0,l.survival(tau,shape,scale),l.cdf(tau,shape,scale)],
      [0,0,1],
      [0.5,0.5,0]
    ])
  times = np.array(
    [
      [0,tau,Et],
      [0,0,rebootCost],
      [5,5,0]
    ])
  return TimeToAbsorbing(probs,times,2)[0]

def constructMatrices(tau, ti, xi = None, rebootCost = 199.997, distr = None):
    p0 = np.zeros(3)
    t0 = np.zeros(3)
    for i in ti:
        if i < tau:
            p0 += np.array([0, 0, 1.0])
            t0 += np.array([0, 0, i*1.0])
        else:
            p0 += np.array([0, 1.0, 0])
            t0 += np.array([0, tau*1.0, 0])
    if xi is not None and distr is not None:
        for x in xi:
            if tau < x:
                p0 += np.array([0, 1.0, 0])
                t0 += np.array([0, tau*1.0, 0])
            else:
                pless = distr.cdf(tau) - distr.cdf(x)
                pmore = distr.survival(tau)
                p0 += np.array([0, pmore/(pmore+pless), pless/(pmore+pless)])
                tless = distr.expectedXBwLts(x,tau) if pless > 1e-6 else 0
                t0 += np.array([0, tau, tless]) * np.array([0, pmore/(pmore+pless), pless/(pmore+pless)])
    t0[1] = t0[1] / p0[1] if p0[1] > 0 else 0
    t0[2] = t0[2] / p0[2] if p0[2] > 0 else 0
    p0 = p0 / sum(p0)
    probs = np.array(
    [
      p0,
      [0,0,1],
      [0.5,0.5,0]
    ])
    times = np.array(
    [
      t0,
      [0,0,rebootCost],
      [5.0,5.0,0]
    ])
    return (probs, times)

def non_parametric(recoveryTimes, currentCensor = 600.0, rebootCost = 200.0, verbose = False):
    '''
    Finds the optimal threshold given some data and making no assumption about the distribution.
    '''
    numOfReboots = sum(recoveryTimes > currentCensor)
    relSavings = []
    taus = np.arange(10, currentCensor, 5)
    indx = 0
    neglosses = []
    poslosses = []
    for tau in taus:
        indx += 1
        savings = numOfReboots * (currentCensor - tau)
        losses = 0
        for i in recoveryTimes:
            if i > tau and i < currentCensor:
                losses = losses + (tau + rebootCost - i)
        netSavings = (savings - losses)
        relSavings.append(netSavings)
        if indx%20 == 0 and verbose:
            print("tau: " + "{0:.2f}".format(tau) + " savings: " + "{0:.2f}".format(savings) + " losses: " + "{0:.2f}".format(losses) + " net: " + "{0:.2f}".format(netSavings))
    return taus[np.argmax(relSavings)]


def compare_non_parametric_approaches(ti):
    costs = []
    for tau in np.arange(10,600,5):
        (p,t) = constructMatrices(tau, ti)
        costs.append(TimeToAbsorbing(p,t,2)[0])
    print("Optimal thresholds based on transition matrices based non parametric approach.")
    print(np.arange(10,600,5)[np.argmin(costs)])
    print("And optimal thresholds from relative savings based non parametric approach.")
    print(non_parametric(ti))
    return costs


def compare_pure_impure_parametric():
    #After some hit and trial for deciding the censor level.
    censorLevel = 70
    [ti1,xi] = censoreData(ti, censorLevel)
    l = Lomax(ti = ti1, xi = xi)
    w = Weibull(ti = ti1, xi = xi)
    ln = Lognormal(ti = ti1, xi = xi)
    costs1 = []
    costs2 = []
    for tau in np.arange(10,300,5):
        (p1,t1) = constructMatrices(tau, ti1, xi, distr = ln)
        costs1.append(TimeToAbsorbing(p1,t1,2)[0] + 0*tau) #Cost for hybrid.
        costs2.append(cost(tau, ln) + 0*tau) #Cost for pure parametric.
    print(np.arange(10,300,5)[np.argmin(costs1)])
    print(np.arange(10,300,5)[np.argmin(costs2)])

    plt.plot(np.arange(10,300,5), costs1)
    plt.axvline(np.arange(10,300,5)[np.argmin(costs1)], color="red")
    plt.show()


def tst():
    t = data_gen_1()
    x = np.ones(7)*800
    ll = Lomax(ti=t, xi=x)
    original = downtime_with_threshold(t, x, ll, True)

    opts = []
    opts1 = []
    opts2 = []
    opts3 = []
    opts4 = []
    logliks = []
    logliks1 = []
    logliks2 = []
    logliks3 = []
    logliks4 = []
    dats = []
    costs = []
    for p in np.arange(0.1, 1.1, 0.1):
        [t2, x2] = censoreData_1(t, x, 220, p)
        dats.append([t2, x2])
        models = [LogLogistic(ti=t2, xi=x2), Lomax(ti=t2, xi=x2), Weibull(ti=t2, xi=x2), Lognormal(ti=t2, xi=x2)]
        liks = [i.loglik(t2, x2, i.params[0], i.params[1]) for i in models]
        #ll = models[np.argmax(liks)]
        #logliks.append(max(liks))
        #opts.append(downtime_with_threshold(t2, x2, ll))
        opts1.append(downtime_with_threshold(t2, x2, models[0])[0])
        opts2.append(downtime_with_threshold(t2, x2, models[1])[0])
        opts3.append(downtime_with_threshold(t2, x2, models[2])[0])
        opts4.append(downtime_with_threshold(t2, x2, models[3])[0])
        costs.append(downtime_with_threshold(t2, x2, models[0])[1])
        logliks1.append(liks[0])
        logliks2.append(liks[1])
        logliks3.append(liks[2])
        logliks4.append(liks[3])

    #plt.plot(np.arange(0.1, 1.1, 0.1), opts)
    #plt.show()
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(np.arange(0.1, 1.1, 0.1), opts1, color = 'orange')
    axarr[0].plot(np.arange(0.1, 1.1, 0.1), opts2, color = 'yellow')
    axarr[0].plot(np.arange(0.1, 1.1, 0.1), opts3, color = 'red')
    axarr[0].plot(np.arange(0.1, 1.1, 0.1), opts4, color = 'green')
    axarr[1].plot(np.arange(0.1, 1.1, 0.1), logliks1, color = 'orange')
    axarr[1].plot(np.arange(0.1, 1.1, 0.1), logliks2, color = 'yellow')
    axarr[1].plot(np.arange(0.1, 1.1, 0.1), logliks3, color = 'red')
    axarr[1].plot(np.arange(0.1, 1.1, 0.1), logliks4, color = 'green')
    plt.show()


def downtime_with_threshold(t, x, ll, plot = False):
    costs = []
    for tau in np.arange(60,500,5):
        (p1,t1) = constructMatrices(tau, ti=t, xi = x, distr = ll)
        costs.append(TimeToAbsorbing(p1,t1,2)[0])
    if plot:
        plt.plot(np.arange(60,500,5), costs)
        plt.axvline(np.arange(60,500,5)[np.argmin(costs)], color="red")
        plt.show()
    return (np.arange(60,500,5)[np.argmin(costs + np.arange(60,500,5)*5e-3)], costs)


if __name__ == '__main__':
    l = Lomax(1.05, 0.1)
    ti = l.samples(size = 10000)
    costs = compare_non_parametric_approaches(ti)
    plt.plot(np.arange(10,600,5), costs)
    plt.axvline(np.arange(10,600,5)[np.argmin(costs)], color="blue")

    print("Now we add some noisy low-duration events.")
    tii = np.copy(ti)
    ti = np.concatenate((tii, np.array([0.1]*10000)), axis = 0)
    costs = compare_non_parametric_approaches(ti)


