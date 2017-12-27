import numpy as np
from LogNormal import *
from Lomax import *
from z_toyTransitions import MCClosedForm

def cost(tau1 = -1, tau2 = -1, tau3 = -1, taus = None):
  if taus is not None:
    [tau1, tau2, tau3] = taus
  
  probs = np.zeros((6,6))
  times = np.zeros((6,6))

  l = Lomax()
  lmb = 0.1
  k = 1.05
  
  ## From Unhealthy
  p1 = 58/563.0 # Power cycle will succeed.
  p2 = 296.0/(296+563+58) # There will be a PXE.
  probs[0,1] = p1 * (1-p2) * l.survival(tau1, k, lmb)
  probs[0,2] = p2 * l.survival(tau1, k, lmb)
  probs[0,3] = (1-p1)*(1-p2)*l.survival(tau1, k, lmb)
  probs[0,5] = l.cdf(tau1,k,lmb) * p2 + l.cdf(tau1, k, lmb)
  times[0,1] = times[0,2] = times[0,3] = tau1
  times[0,5] = l.expectedT(tau1,k,lmb) # If PXE and non-PXE distributions are different then this will be a weighted sum.

  ## From Booting
  p3 = 36.0/(36+56)
  probs[2,1] = (1-p3) * l.survival(tau3, k, lmb)
  probs[2,4] = p3 * l.survival(tau3, k, lmb)
  probs[2,5] = l.cdf(tau3,k,lmb)
  times[2,1] = tau3
  times[2,4] = 3 * tau3 - 3 # The subtraction by three is the average value of the beta distribution.
  times[2,5] = l.expectedT(tau3,k,lmb) + 3 # The 3 here is the expected time between unhealthy to PXE.

  ## From PoweringOn
  probs[1,3] = l.survival(tau2, k, lmb)
  probs[1,5] = l.cdf(tau2,k,lmb)
  times[1,3] = tau2
  times[1,5] = l.expectedT(tau2,k,lmb)

  probs[3,5] = 1.0
  times[3,5] = 400

  probs[4,3] = 1.0
  times[4,3] = 10

  return MCClosedForm(probs,times,5)[0]

def grad(taus = np.array([300, 300, 300])):
  [tau1, tau2, tau3] = taus
  delTau1 = (cost(taus[0]+1e-5,taus[1],taus[2]) - cost(taus[0]-1e-5,taus[1],taus[2]))/1e-5/2
  delTau2 = (cost(taus[0],taus[1]+1e-5,taus[2]) - cost(taus[0],taus[1]-1e-5,taus[2]))/1e-5/2
  delTau3 = (cost(taus[0],taus[1],taus[2]+1e-5) - cost(taus[0],taus[1],taus[2]-1e-5))/1e-5/2
  return np.array([delTau1, delTau2, delTau3])

def hess(taus = np.array([300, 300, 300])):
  eps = 1e-4
  deltau1sq = (cost(taus[0]+2*eps,taus[1],taus[2]) + cost(taus[0]-2*eps,taus[1], taus[2]) - 2*cost(taus = taus))/4/eps/eps
  deltau2sq = (cost(taus[0],taus[1]+2*eps,taus[2]) + cost(taus[0],taus[1]-2*eps, taus[2]) - 2*cost(taus = taus))/4/eps/eps
  deltau3sq = (cost(taus[0],taus[1],taus[2]+2*eps) + cost(taus[0],taus[1],taus[2]-2*eps) - 2*cost(taus = taus))/4/eps/eps
  deltau1tau2 = (cost(taus[0]+eps,taus[1]+eps,taus[2]) + cost(taus[0]-eps,taus[1]-eps,taus[2]) - cost(taus[0]+eps,taus[1]-eps,taus[2]) - cost(taus[0]-eps,taus[1]+eps,taus[2]))/4/eps/eps
  deltau1tau3 = (cost(taus[0]+eps,taus[1],taus[2]+eps) + cost(taus[0]-eps,taus[1],taus[2]-eps) - cost(taus[0]+eps,taus[1],taus[2]-eps) - cost(taus[0]-eps,taus[1],taus[2]+eps))/4/eps/eps
  deltau2tau3 = (cost(taus[0],taus[1]+eps,taus[2]+eps) + cost(taus[0],taus[1]-eps,taus[2]-eps) - cost(taus[0],taus[1]+eps,taus[2]-eps) - cost(taus[0],taus[1]-eps,taus[2]+eps))/4/eps/eps
  hess = np.zeros([3,3])
  hess[0,0] = deltau1sq
  hess[1,1] = deltau2sq
  hess[2,2] = deltau3sq
  hess[0,1] = hess[1,0] = deltau1tau2
  hess[0,2] = hess[2,0] = deltau1tau3
  hess[1,2] = hess[2,1] = deltau2tau3
  return hess

def newtonRh(numIter=7001, params = np.array([100,100,100])):
  steps = {1.0:0, 2.0:0, 2.5:0, 3.0:0, 3.5:0, 3.7:0, 4.0:0, 4.5:0, 4.7:0, 5.5:0, 6.0:0, 6.5:0, 7.0:0, 7.5:0, 8.0:0, 8.5:0, 9.0:0, 9.5:0, 10.0:0, 12.0:0, 15.0:0, 20.0:0, 25.0:0, 27.0:0, 35.0:0, 37.0:0, 40.0:0, 50.0:0,100.0:0,200.0:0,500.0:0,1000.0:0}
  for i in xrange(numIter):
      directn = grad(params)
      if sum(abs(directn)) < 1e-5:
          print "\nIt took: " + str(i) + " Iterations.\n Gradients - " + str(directn)
          return params
      lik = cost(taus=params)
      #hessian = np.eye(3)
      hessian = hess(params)
      try:
        step = np.linalg.solve(hessian, directn)
      except:
        step = directn
      params2 = params - 1e-6 * step
      for alp1 in steps.keys():
        params1 = params - alp1 * step
        try:
            lik1 = cost(taus = params1)
            if lik1 < lik and np.isfinite(lik1):
              lik = lik1
              params2 = params1
        except:
            continue
      params = params2
      #if i%10 == 0:
      print "Iteration " + str(i) + " ,objective function: " + str(lik) + " \nparams = " + str(params) + " \nGradient = " + str(directn) + "\n##\n\n"
  return params

'''
Given the transition probabilities and transition times matrices, outputs the proportion of time spent in 
each state via simulation and closed form. The two should always be close.
'''
def SteadyStateMonteCarloValidation(
  p = np.matrix([
          [0,.2,.4,.4],
          [.2,0,.4,.4],
          [.2,.3,0,.5],
          [.3,.4,.3,0]
          ]),
    t = np.matrix([
          [2,2,2,1],
          [1,1,5,1],
          [1,1,1,1],
          [1,1,1,1]
          ]),
    starting_state = 2
):
  states = np.zeros(p.shape[0])
  states_w_times = np.zeros(p.shape[0])
  curr_state = starting_state
  for i in range(10000):
    next_state = np.random.choice(p.shape[0], p=np.array(p[curr_state,])[0])
    states[curr_state] = states[curr_state] + 1
    states_w_times[curr_state] = states_w_times[curr_state] + t[curr_state, next_state]
    curr_state = next_state
  p_times_t = np.array(np.sum(np.multiply(p,t),axis=1).T)[0]
  # Solve for pi by finding the null space of (P-I)
  pis = np.linalg.svd(p-np.eye(p.shape[0]))[0][:,p.shape[0]-1]
  pis = pis/sum(pis)
  res = [states_w_times/sum(states_w_times), pis, p_times_t]
  props1 = np.multiply(res[1].T,res[2])/sum(np.multiply(res[1].T,res[2]).T)
  props2 = res[0]
  return [props1, props2]


def SteadyStateMonteCarlo(
  p = np.matrix([
          [0,.2,.4,.4],
          [.2,0,.4,.4],
          [.2,.3,0,.5],
          [.3,.4,.3,0]
          ]),
    t = np.matrix([
          [2,2,2,1],
          [1,1,5,1],
          [1,1,1,1],
          [1,1,1,1]
          ])
    ):
    p_times_t = np.array(np.sum(np.multiply(p,t),axis=1).T)[0]
    pis = np.linalg.svd(p-np.eye(p.shape[0]))[0][:,p.shape[0]-1]
    pis = pis/sum(pis)
    res = [0.0, pis, p_times_t]
    props1 = np.multiply(res[1].T,res[2])/sum(np.multiply(res[1].T,res[2]).T)
    return props1


