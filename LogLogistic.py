import numpy as np
from Sigmoid import *
from BaseModel import *

class Loglogistic(Base):
    def __init__(self,alp=1, beta=0.5):
        self.train = []
        self.test = []
        self.train_org = []
        self.train_inorg = []
        self.alp = alp
        self.beta = beta
        self.params = []

    def pdf(self,x,alp,beta):
        return (beta/alp)*(x/alp)**(beta-1)/(1+(x/alp)**beta)**2

    def cdf(self,x,alp,beta):
        return 1/(1+(x/alp)**-beta)

    def logpdf(self,x,alp,beta):
        return np.log(beta)-np.log(alp) +(beta-1)*(np.log(x) - np.log(alp)) - 2*np.log(1+(x/alp)**beta)

    def survival(self,x,alp,beta):
        return 1-self.cdf(x,alp,beta)

    def logsurvival(self,x,alp,beta):
        return np.log(self.survival(x,alp,beta))

    def loglik(self,t,x,alp,beta):
        return sum(self.logpdf(t,alp,beta)) + sum(self.logsurvival(x,alp,beta))

    def grad(self,t,x,alp,beta):
        n = len(t)
        m = len(x)
        delalp = -n*beta/alp +2*beta/alp**(beta+1) * sum(t**beta/(1+(t/alp)**beta)) + beta/alp**(beta+1)*sum(x**beta/(1+(x/alp)**beta))
        delbeta = n/beta -n*np.log(alp) + sum(np.log(t)) -2*sum((t/alp)**beta/(1+(t/alp)**beta)*np.log(t/alp) ) - sum((x/alp)**beta/(1+(x/alp)**beta)*np.log(x/alp))
        return np.array([delalp,delbeta])

    def numerical_grad(self,t,x,alp,beta):
        eps = 1e-5
        delalp = (self.loglik(t,x,alp+eps,beta) - self.loglik(t,x,alp-eps,beta))/2/eps
        delbeta = (self.loglik(t,x,alp,beta+eps) - self.loglik(t,x,alp,beta-eps))/2/eps
        return np.array([delalp,delbeta])

    def gradient_descent(self, numIter=2001, params = np.array([2.0,2.0])):
        for i in xrange(numIter):
            #lik = self.loglik(self.train_org,self.train_inorg,params[0],params[1],params[2])
            directn = self.grad(self.train_org,self.train_inorg,params[0],params[1])
            params2 = params + 1e-10*directn
            lik = self.loglik(self.train_org,self.train_inorg,params2[0],params2[1])
            for alp1 in [1e-8,1e-7,1e-5,1e-3,1e-2,.1]:
                params1 = params + alp1 * directn
                if(min(params1) > 0):
                    lik1 = self.loglik(self.train_org,self.train_inorg,params1[0],params1[1])
                    if(lik1 > lik and np.isfinite(lik1)):
                        lik = lik1
                        params2 = params1
            params = params2
            if i%100 == 0:
                print "Iteration " + str(i) + " ,objective function: " + str(lik) + " \nparams = " + str(params) + " \nGradient = " + str(directn)
                print "\n########\n"
        [self.mu,self.sigma] = params
        self.params = params
        return params

