import numpy as np

class ModifiedWeibull():
    def __init__(self):
        self.train = []
        self.test = []
        self.train_org = []
        self.train_inorg = []
        self.alp = 0
        self.beta = 0
        self.gamma = 0

    def cdf(self,x,alp=0.5,beta=0.2,gamma=0.1):
        return 1-np.exp(-alp*x-beta*x**gamma)

    def survival(self,x,alp=0.5,beta=0.2,gamma=0.1):
        return np.exp(-alp*x-beta*x**gamma)

    def pdf(self,x,alp=0.5,beta=0.2,gamma=0.1):
        return (alp+beta*gamma*x**(gamma-1))*np.exp(-alp*x-beta*x**gamma)

    def hazard(self,x,alp=0.5,beta=0.2,gamma=0.1):
        return alp+beta*gamma*x**(gamma-1)

    def logpdf(self,x,alp=0.5,beta=0.2,gamma=0.1):
        return np.log(alp+beta*gamma*x**(gamma-1)) - alp*x-beta*x**gamma

    def logsurvival(self,x,alp=0.5,beta=0.2,gamma=0.1):
        return -alp*x-beta*x**gamma

    def loglik(self,t,x=None,alp=0.5,beta=0.2,gamma=0.1):
        if x == None:
            return sum(self.logpdf(t,alp,beta,gamma))
        else:
            return sum(self.logpdf(t,alp,beta,gamma)) + sum(self.logsurvival(x,alp,beta,gamma))

    def grad(self,t,x=None,alp=0.5,beta=0.2,gamma=0.1):
        delalp = sum(1/(alp+beta*gamma*t**(gamma-1))) - sum(t) - sum(x)
        delbeta = sum(gamma*t**(gamma-1)/(alp+beta*gamma*t**(gamma-1))) - sum(t**gamma) - sum(x**gamma)
        delgamma = sum(beta*t**(gamma-1)*(1+gamma*np.log(t))/(alp+beta*gamma*t**(gamma-1)) ) - beta*sum(t**gamma*np.log(t)) - beta*sum(x**gamma*np.log(x)) 
        return np.array([delalp,delbeta,delgamma])

    def numerical_grad(self,t,x=None,alp=0.5,beta=0.2,gamma=0.1):
        eps = 1e-5
        delalp = (self.loglik(t,x,alp+eps,beta,gamma)-self.loglik(t,x,alp-eps,beta,gamma))/2/eps
        delbeta = (self.loglik(t,x,alp,beta+eps,gamma)-self.loglik(t,x,alp,beta-eps,gamma))/2/eps
        delgamma = (self.loglik(t,x,alp,beta,gamma+eps)-self.loglik(t,x,alp,beta,gamma-eps))/2/eps
        return np.array([delalp,delbeta,delgamma])

    def gradient_descent(self, numIter=2001, params = np.array([.5,.3,.2])):
        for i in xrange(numIter):
            #lik = self.loglik(self.train_org,self.train_inorg,params[0],params[1],params[2])
            directn = self.grad(self.train_org,self.train_inorg,params[0],params[1],params[2])
            params2 = params + 1e-9*directn
            lik = self.loglik(self.train_org,self.train_inorg,params2[0],params2[1],params2[2])
            for alp1 in [1e-8,1e-7,1e-5,1e-3,1e-2,.1]:
                params1 = params + alp1 * directn
                if(min(params1) > 0):
                    lik1 = self.loglik(self.train_org,self.train_inorg,params1[0],params1[1],params1[2])
                    if(lik1 > lik and np.isfinite(lik1)):
                        lik = lik1
                        params2 = params1
            params = params2
            if i%100 == 0:
                print "Iteration " + str(i) + " ,objective function: " + str(lik) + " \nparams = " + str(params) + " \nGradient = " + str(directn)
                print "\n########\n"
        return params

# -79342.1249811


'''
Iteration 73300 ,objective function: -68193.6048994
params = [-0.09227194  0.11365995  0.97235938]
Gradient = [ 20240.11875322  17940.8823032   12970.68861657]
'''


'''
Iteration 73100 ,objective function: -68193.4995454
params = [-0.09217072  0.11354001  0.97232306]
Gradient = [   37.06932588  1055.02719928   516.17510825]
'''
