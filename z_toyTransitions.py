'''
Wanted to test how full-fledged optimization would stack against piece-wise optimization.
'''

import numpy as np
from LogNormal import *
from Lomax import *

## Single parameter cost function for validation.
def cost1(tau):
  l = Lomax()
  lmb = 0.1
  k = 1.05
  Et = l.expectedT(tau,k,lmb)
  rebootCost = 199.997
  probs = np.array(
    [
      [0,l.survival(tau,k,lmb),l.cdf(tau,k,lmb)],
      [0,0,1],
      [0,0,0]
    ])
  times = np.array(
    [
      [0,tau,Et],
      [0,0,rebootCost],
      [0,0,0]
    ])
  return MCClosedForm(probs,times,2)[0]


def cost(tau1 = -1, tau2 = -1,
        taus = None):
  if taus is not None:
    [tau1, tau2] = taus
  l = Lomax()
  lmb = 0.1
  k = 1.05
  p = 0.2
  hiCost = 400
  rebootCost = l.expectedDT(tau2,k,lmb,hiCost) # Replace lomax with log logistic here.
  Et_1 = l.expectedT(tau1,k,lmb)
  Et_2 = l.expectedT(tau2,k,lmb)
  probs = np.array(
  		[
  			[0,l.survival(tau1,k,lmb),0,l.cdf(tau1,k,lmb)],
  			[p,0, (1-p)*l.survival(tau2,k,lmb) , (1-p)*l.cdf(tau2,k,lmb)],
  			[0,0,0,1],
  			[0,0,0,0]
  		])
  times = np.array(
  		[
  			[0, tau1,0,Et_1],
  			[200,0,tau2,Et_2],
  			[0,0,0,hiCost],
  			[0,0,0,0]
  		])
  return MCClosedForm(probs,times,3)[0]

def grad(taus = np.array([300,300])):
  tau1 = taus[0]
  tau2 = taus[1]
  delTau1 = (cost(taus[0]+1e-5,taus[1]) - cost(taus[0]-1e-5,taus[1]))/1e-5/2
  delTau2 = (cost(taus[0],taus[1]+1e-5) - cost(taus[0],taus[1]-1e-5))/1e-5/2
  return np.array([delTau1, delTau2])


def hess(taus = np.array([300,300])):
  eps = 1e-4
  deltau1sq = (cost(k+2*eps,lmb) + cost(k-2*eps,lmb) - 2*cost(k,lmb))/4/eps/eps
  deltau2sq = (cost(k,lmb+2*eps) + cost(k,lmb-2*eps) - 2*cost(k,lmb))/4/eps/eps
  deltau1tau2 = (cost(k+eps,lmb+eps) + cost(k-eps,lmb-eps) - cost(k+eps,lmb-eps) - cost(k-eps,lmb+eps))/4/eps/eps
  hess = np.zeros([2,2])
  hess[0,0] = deltau1sq
  hess[1,1] = deltau2sq
  hess[0,1] = hess[1,0] = deltau1tau2
  return hess

def newtonRh(numIter=1001, params = np.array([100,100])):
  for i in xrange(numIter):
      directn = grad(params)
      if sum(abs(directn)) < 1e-5:
          print "\nIt took: " + str(i) + " Iterations.\n Gradients - " + str(directn)
          return params
      lik = cost(taus=params)
      step = np.linalg.solve(hess(params),directn)
      params = params - step
      if i%10 == 0:
          print "Iteration " + str(i) + " ,objective function: " + str(lik) + " \nparams = " + str(params) + " \nGradient = " + str(directn) + "\n##\n\n"
  return params


state_indices = ["Raw","Ready","Unhealthy","Booting","PoweringOn","Dead","HumanInvestigate","Recovering"]
def MCClosedForm(
        p = np.matrix([
              [0,.2,.4,.3,.1,0,0],
              [.2,0,.3,.4,0,.1,0],
              [.1,.3,0,.5,0,.1,0],
              [.2,.3,.2,0,.1,.1,.1],
              [.2,.2,0,.5,0,.1,0],
              [.2,.3,0,.1,0,0,.4],
              [.2,.1,.2,.3,0,.2,0]
              ]),
        t = np.matrix([
              [1,2,3,1,1,3,2],
              [1,2,3,1,1,3,2],
              [1,2,3,1,1,3,2],
              [1,2,3,1,1,3,2],
              [1,2,3,1,1,3,2],
              [1,2,3,1,1,3,2],
              [1,2,3,1,1,3,2]
              ]),
        absorbing_state = 1
              ):
    rhs = np.diag(np.dot(p, t.T))
    rhs = np.delete(rhs,absorbing_state)
    q = np.delete(p, np.s_[absorbing_state],1)
    q = np.delete(q, np.s_[absorbing_state],0)
    lhs = (np.eye(q.shape[0]) - q)
    x = np.linalg.solve(lhs,rhs)
    return np.insert(x,absorbing_state,0)


