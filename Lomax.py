import numpy as np
from scipy.stats import lomax

class Lomax():
    def __init__(self):
        self.train = []
        self.test = []
        self.train_org = []
        self.train_inorg = []
        self.k = 0
        self.lmb = 0
        self.params = []

    def pdf(self,t,k,lmb):
        return lmb*k/(1+lmb*t)**(k+1)

    def cdf(self,t,k,lmb):
        return 1-(1+lmb*t)**-k

    def survival(self,t,k,lmb):
        return (1+lmb*t)**-k

    def logpdf(self,t,k,lmb):
        return np.log(k) + np.log(lmb) - (k+1)*np.log(1+lmb*t)

    def logsurvival(self,t,k,lmb):
        return -k*np.log(1+lmb*t)

    def loglik(self,t,x,k=0.5,lmb=0.3):
        return sum(self.logpdf(t,k,lmb)) +sum(self.logsurvival(x,k,lmb))

    def grad(self,t,x,k=0.5,lmb=0.3):
        n = len(t)
        m = len(x)
        delk = n/k - sum(np.log(1+lmb*t)) - sum(np.log(1+lmb*x))
        dellmb = n/lmb -(k+1) * sum(t/(1+lmb*t)) -k*sum(x/(1+lmb*x))
        return np.array([delk,dellmb])
    
    def numerical_grad(self,t,x,k=0.5,lmb=0.3):
        eps = 1e-5
        delk = (self.loglik(t,x,k+eps,lmb) - self.loglik(t,x,k-eps,lmb))/2/eps
        dellmb = (self.loglik(t,x,k,lmb+eps) - self.loglik(t,x,k,lmb-eps))/2/eps
        return np.array([delk,dellmb])

    def hessian(self,t,x,k=0.5,lmb=0.3):
        n=len(t)
        delksq = -n/k**2
        dellmbsq = -n/lmb**2 + (k+1)*sum((t/(1+lmb*t))**2) + k*sum((x/(1+lmb*x))**2)
        delklmb = -sum(t/(1+lmb*t)) - sum(x/(1+lmb*x))
        hess = np.zeros([2,2])
        hess[0,0] = delksq
        hess[1,1] = dellmbsq
        hess[0,1] = hess[1,0] = delklmb
        return hess

    def numerical_hessian(self,t,x,k=0.5,lmb=0.3):
        eps = 1e-4
        delksq = (self.loglik(t,x,k+2*eps,lmb) + self.loglik(t,x,k-2*eps,lmb) - 2*self.loglik(t,x,k,lmb))/4/eps/eps
        dellmbsq = (self.loglik(t,x,k,lmb+2*eps) + self.loglik(t,x,k,lmb-2*eps) - 2*self.loglik(t,x,k,lmb))/4/eps/eps
        dellmbk = (self.loglik(t,x,k+eps,lmb+eps) + self.loglik(t,x,k-eps,lmb-eps) - self.loglik(t,x,k+eps,lmb-eps) - self.loglik(t,x,k-eps,lmb+eps))/4/eps/eps
        hess = np.zeros([2,2])
        hess[0,0] = delksq
        hess[1,1] = dellmbsq
        hess[0,1] = hess[1,0] = dellmbk
        return hess

    def gradient_descent(self,numIter=2001, params = np.array([.5,.3])):
        for i in xrange(numIter):
            lik = self.loglik(self.train_org,self.train_inorg,params[0],params[1])
            directn = self.grad(self.train_org,self.train_inorg,params[0],params[1])
            params2 = params
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
        return params

    def newtonRh(self,numIter=101, params = np.array([.1,.1])):
        for i in xrange(numIter):
            directn = self.grad(self.train_org,self.train_inorg,params[0],params[1])
            if sum(abs(directn)) < 1e-5:
                print "\nIt took: " + str(i) + " Iterations.\n Gradients - " + str(directn)
                self.params = params
                [self.k,self.lmb] = params
                return params
            lik = self.loglik(self.train_org,self.train_inorg,params[0],params[1])
            step = np.linalg.solve(self.hessian(self.train_org,self.train_inorg,params[0],params[1]),directn)
            params = params - step
            if min(params) < 0:
                print "Drastic measures"
                params = params + step # undo the effect of taking the step.
                for alp1 in [1e-8,1e-7,1e-5,1e-3,1e-2,.1,.5,1.0]:
                    params1 = params - alp1 * step
                    if(max(params1) > 0):
                        lik1 = self.loglik(self.train_org,self.train_inorg,params1[0],params1[1])
                        if(lik1 > lik and np.isfinite(lik1)):
                            lik = lik1
                            params2 = params1
                            scale = alp1
                params = params2
            if i%10 == 0:
                print "Iteration " + str(i) + " ,objective function: " + str(lik) + " \nparams = " + str(params) + " \nGradient = " + str(directn) + "\n##\n\n"
        [self.k,self.lmb] = params
        self.params = params
        return params

    def samples(self, k, lmb, size = 1000):
        return lomax.rvs(c=k, scale=(1 / lmb),size=size)

    def kappafn_k(self,t,x,lmb=0.1):
        n = len(t)
        return n/(sum(np.log(1+lmb*t)) + sum(np.log(1+lmb*x)))

    def kappafn_lmb(self,t,x,lmb=0.1):
        n = len(t)
        return (n/lmb - sum(t/(1+lmb*t)))/(sum(t/(1+lmb*t)) + sum(x/(1+lmb*x)))

    def bisection_fn(self,lmb=0.1):
        return self.kappafn_k(self.train_org,self.train_inorg,lmb) - self.kappafn_lmb(self.train_org,self.train_inorg,lmb)

    def bisection(self,a=1e-6,b=2000):
        n=1
        while n<10000:
            c=(a+b)/2
            if self.bisection_fn(c)==0 or (b-a)/2<1e-6:
                return c
            n=n+1
            if (self.bisection_fn(c) > 0) == (self.bisection_fn(a) > 0):
                a=c
            else:
                b=c


# Optimal likelihood for test dataset:
# -90,527.829
# Optimal tau with C_int circa 480: 