import numpy as np
from BaseModel import *

class ExponentialWeibull(Base):
    def __init__(self,lmb=0.43947146,alp=0.38530997,theta=0.6339891):
        self.train = []
        self.test = []
        self.train_org = []
        self.train_inorg = []
        self.lmb = lmb
        self.alp = alp
        self.theta = theta

    def determine_params(self,lmb,alp,theta,params):
        if params is not None:
            lmb = params[0]
            alp = params[1]
            theta = params[2]
        else:
            if lmb == -1:
                lmb = self.lmb
            if alp == -1:
                alp = self.alp
            if theta == -1:
                theta = self.theta
        return [lmb,alp,theta]

    def pdf(self,t,lmb=-1,alp=-1,theta=-1,params=None):
        [lmb,alp,theta] = self.determine_params(lmb,alp,theta,params)
        return lmb*alp*theta*t**(theta-1)*np.exp(-alp*t**theta)*(1-np.exp(-alp*t**theta))**(lmb-1)

    def survival(self,t,lmb=-1,alp=-1,theta=-1,params=None):
        [lmb,alp,theta] = self.determine_params(lmb,alp,theta,params)
        return 1-(1-np.exp(-alp*t**theta))**lmb

    def logpdf(self,t,lmb=0.5,alp=0.3,theta=0.2):
        return np.log(lmb) + np.log(alp) + np.log(theta) + (theta-1)*np.log(t) - alp*t**theta + (lmb-1)*np.log((1-np.exp(-alp*t**theta)))

    def logsurvival(self,t,lmb=0.5,alp=0.3,theta=0.2):
        return np.log( 1-(1-np.exp(-alp*t**theta))**lmb )

    def loglik(self,t,x,lmb=0.5,alp=0.3,theta=0.2):
        return sum(self.logpdf(t,lmb,alp,theta)) + sum(self.logsurvival(x,lmb,alp,theta))

    def hazard(self,t,lmb=0.43947146,alp=0.38530997,theta=0.6339891):
        return self.pdf(t,lmb,alp,theta)/self.survival(t,lmb,alp,theta)

    def grad(self,t,x,lmb=0.5,alp=0.3,theta=0.2):
        n = len(t)
        survival = 1 - (1 - np.exp(-alp*x**theta))**lmb
        delalp = n/alp - sum(t**theta) + (lmb-1) * sum(t**theta/(np.exp(alp*t**theta)-1)) -lmb*sum( (1-np.exp(-alp*x**theta))**(lmb-1)*np.exp(-alp*x**theta)*x**theta/(survival) )
        deltheta = n/theta + sum(np.log(t)) -alp*sum(np.log(t)*t**theta) + alp*(lmb-1)*sum(np.log(t)*t**theta/(np.exp(alp*t**theta)-1)) - lmb*alp*sum((1-np.exp(-alp*x**theta))**(lmb-1)*np.exp(-alp*x**theta)*x**theta*np.log(x)/survival )
        dellmb = n/lmb + sum(np.log(1-np.exp(-alp*t**theta))) - sum((1-np.exp(-alp*x**theta))**lmb*np.log(1-np.exp(-alp*x**theta))/(survival) )
        return np.array([dellmb, delalp, deltheta])

    def numerical_grad(self,t,x,lmb=0.5,alp=0.3,theta=0.2):
        eps = 1e-5
        dellmb = (self.loglik(t,x,lmb+eps,alp,theta)-self.loglik(t,x,lmb-eps,alp,theta))/2/eps
        delalp = (self.loglik(t,x,lmb,alp+eps,theta)-self.loglik(t,x,lmb,alp-eps,theta))/2/eps
        deltheta = (self.loglik(t,x,lmb,alp,theta+eps)-self.loglik(t,x,lmb,alp,theta-eps))/2/eps
        return np.array([dellmb, delalp, deltheta])

    def gradient_descent(self,numIter=2001, params = np.array([.1,.1,.1])):
        [t,x] = [self.train_org,self.train_inorg]
        for i in xrange(numIter):
            directn = self.grad(t,x,params[0],params[1],params[2])
            params2 = params + 1e-9*directn
            lik = self.loglik(t,x,params2[0],params2[1],params2[2])
            for alp1 in [1e-8,1e-7,1e-5,1e-3,1e-2,.1]:
                params1 = params + alp1 * directn
                if(max(params1) > 0):
                    lik1 = self.loglik(t,x,params1[0],params1[1],params1[2])
                    if(lik1 > lik and np.isfinite(lik1)):
                        lik = lik1
                        params2 = params1
            params = params2
            if i%100 == 0:
                print "Iteration " + str(i) + " ,objective function: " + str(lik) + " \nparams = " + str(params) + " \nGradient = " + str(directn)
                print "\n########\n"
        [self.lmb,self.alp,self.theta] = params
        return params

if __name__ == '__main__':
    e = ExponentialWeibull()
    t = e.train_org
    x = e.train_inorg

# Likelihood on test data: -79,118.5976


