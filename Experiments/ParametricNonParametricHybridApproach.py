import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from Weibull import *
from Lomax import *
from LogNormal import *
from LogLogistic import *
from MarkovChains import *

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


def censoreData(ti, censorLevel):
    #xi = np.ones(sum(ti >= censorLevel)) * censorLevel
    #ti1 = ti[ti < censorLevel]
    xi = []
    ti1 = []
    for i in ti:
        if i > censorLevel and np.random.uniform() < 0.2:
            xi.append(censorLevel)
        else:
            ti1.append(i)
    return [np.array(ti1), np.array(xi)]


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

if __name__ == '__main__':

l = Lomax(1.05, 0.1)
ti = l.samples(size = 10000)
costs = compare_non_parametric_approaches()
plt.plot(np.arange(10,600,5), costs)
plt.axvline(np.arange(10,600,5)[np.argmin(costs)], color="blue")


print("Now we add some noisy low-duration events.")
tii = np.copy(ti)
ti = np.concatenate((tii, np.array([0.1]*10000)), axis = 0)


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
    costs1.append(TimeToAbsorbing(p1,t1,2)[0] + 0*tau)
    costs2.append(cost(tau, ln) + 0*tau)
print(np.arange(10,300,5)[np.argmin(costs1)])
print(np.arange(10,300,5)[np.argmin(costs2)])

plt.plot(np.arange(10,300,5), costs1)
plt.axvline(np.arange(10,300,5)[np.argmin(costs1)], color="red")


