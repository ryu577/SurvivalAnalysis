'''
Wanted to test how full-fledged optimization would stack against piece-wise optimization.
'''
import numpy as np
from LogNormal import *
from Lomax import *
import matplotlib.pyplot as plt
from scipy.interpolate import spline

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
        taus = None, p = 0.0, indx = 0):
  if taus is not None:
    [tau1, tau2] = taus
  l = Lomax()
  lmb = 0.1
  k = 1.05
  hiCost = 400
  rebootCost = l.expectedDT(tau2,k,lmb,hiCost) # Replace lomax with log logistic here.
  Et_1 = l.expectedT(tau1,k,lmb)
  Et_2 = l.expectedT(tau2,k,lmb)
  # Unhealthy, PoweringOn, HI, Ready
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
  			[50,0,0,hiCost], # 50 is a dummy number added.
  			[0,0,0,0]
  		])
  return MCClosedForm(probs,times,3)[indx]

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
  steps = {1.0:0, 2.0:0, 2.5:0, 3.0:0, 3.5:0, 3.7:0, 4.0:0, 4.5:0, 4.7:0, 5.5:0, 6.0:0, 6.5:0, 7.0:0, 7.5:0, 8.0:0, 8.5:0, 9.0:0, 9.5:0, 10.0:0, 12.0:0, 15.0:0, 20.0:0, 25.0:0, 27.0:0, 35.0:0, 37.0:0, 40.0:0, 50.0:0,100.0:0,200.0:0,500.0:0,1000.0:0}
  for i in xrange(numIter):
      directn = grad(params)
      if sum(abs(directn)) < 1e-5:
          print "\nIt took: " + str(i) + " Iterations.\n Gradients - " + str(directn)
          return params
      lik = cost(taus=params)
      #hessian = hess(params)
      hessian = np.eye(2)
      step = np.linalg.solve(hessian,directn)
      params2 = params - 1e-6 * step
      for alp1 in steps.keys():
        params1 = params - alp1 * step
        lik1 = cost(taus=params1)
        if lik1 < lik and np.isfinite(lik1):
          lik = lik1
          params2 = params1
      params = params2
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
    return np.insert(x, absorbing_state, 0)

def checkCombinedGrad():
  hiCost = 400.0
  optimalTau1 = k * hiCost- 1 / lmb
  optimalTau = k * l.expectedDT(optimalTau1, k, lmb, 400.0) - 1 / lmb
  vec = np.array([optimalTau, optimalTau1])
  grad1 = grad(vec)
  print grad1
  return vec


def plot_smooth(plt, T = np.array([6, 7, 8, 9, 10, 11, 12]), power = np.array([1.53E+03, 5.92E+02, 2.04E+02, 7.24E+01, 2.72E+01, 1.10E+01, 4.70E+00])):
  xnew = np.linspace(T.min(),T.max(),300) #300 represents number of points to make between T.min and T.max
  power_smooth = spline(T,power,xnew)
  plt.plot(xnew, power_smooth, alpha = 0.3)

'''
colors = 'rgb'
utaus = np.arange(10, 900, 2)
j = 0
for p in np.arange(0, 1e-4, 1e-5):
  costs = [ cost(taus=np.array([i,900]),p = p,indx = 1) for i in utaus]
  rgb = heat_rgb(0,1e-4,p)
  ss = sum(rgb)
  rgb = (rgb[0]*1.0/ss, rgb[1]*1.0/ss, rgb[2]*1.0/ss)
  plt.plot(utaus, costs, alpha = 0.5, color = rgb, label = 'p=' + str(p))
  j = j + 1

plt.legend()
plt.xlabel('Unhealthy thresholds (tau)')
plt.ylabel('Intervention costs (Cint)')
'''
