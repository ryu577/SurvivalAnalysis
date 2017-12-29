
import sys
sys.path.append('../')
from Weibull import *
from Lomax import *
from LogNormal import *
from LogLogistic import *
from MarkovChains import *

## Single parameter cost function for validation.
def cost(tau, scale = 0.1, shape = 1.05, rebootCost = 199.997):
  l = Lomax()
  Et = l.expectedT(tau, shape, scale)
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


if __name__ == '__main__':
    l = Lomax(1.05, 0.1)
    ti = l.samples(size = 10000)
    costs = []
    for tau in np.arange(10,300,5):
        (p,t) = constructMatrices(tau, ti)
        costs.append(TimeToAbsorbing(p,t,2)[0])
    print(np.arange(10,300,5)[np.argmin(costs)])

    #After some hit and trial.
    censorLevel = 70
    xi = np.ones(sum(ti >= censorLevel)) * censorLevel
    ti1 = ti[ti < censorLevel]
    l = Lomax(ti = ti1, xi = xi)
    costs = []
    for tau in np.arange(10,300,5):
        (p,t) = constructMatrices(tau, ti1, xi, distr = l)
        costs.append(TimeToAbsorbing(p,t,2)[0])
    print(np.arange(10,300,5)[np.argmin(costs)])


