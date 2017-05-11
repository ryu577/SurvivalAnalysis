import numpy as np

class AdditiveWeibull():
    def __init__(self):
        self.train = []
        self.test = []
        self.train_org = []
        self.train_inorg = []
        self.alp = 0.40398382
        self.theta = 0.26421151
        self.beta = 0.0541794
        self.gamma = 0.73154069

    def cdf(self,t,alp=0.5,theta=0.3,beta=0.2,gamma=0.1):
        return 1-np.exp(-alp*t**beta-beta*t**gamma)

    def pdf(self,t,alp=0.5,theta=0.3,beta=0.2,gamma=0.1):
        return (alp*theta*t**(theta-1)+beta*gamma*t**(gamma-1))*np.exp(-alp*t**theta-beta*t**gamma)

    def logpdf(self,t,alp=0.5,theta=0.3,beta=0.2,gamma=0.1):
        return np.log(alp*theta*t**(theta-1)+beta*gamma*t**(gamma-1)) - (alp*t**theta+beta*t**gamma)

    def logsurvival(self,t,alp=0.5,theta=0.3,beta=0.2,gamma=0.1):
        return -alp*t**theta - beta*t**gamma

    def hazard(self,t,alp=0.40398382,theta=0.26421151,beta=0.0541794,gamma=0.73154069):
        return alp*theta*t**(theta-1) + beta*gamma*t**(gamma-1)

    def loglik(self,t,x=None,alp=0.5,theta=0.3,beta=0.2,gamma=0.1):
        if x == None:
            return sum(self.logpdf(t,alp,theta,beta,gamma))
        else:
            return sum(self.logpdf(t,alp,theta,beta,gamma)) + sum(self.logsurvival(x,alp,theta,beta,gamma))

    def grad(self,t,x=None,alp=0.5,theta=0.3,beta=0.2,gamma=0.1):
        delalp = sum(theta*t**(theta-1)/(alp*theta*t**(theta-1) + beta*gamma*t**(gamma-1))) - sum(t**theta)
        delbeta = sum( gamma*t**(gamma-1)/(alp*theta*t**(theta-1) + beta*gamma*t**(gamma-1))  ) - sum(t**gamma)
        deltheta = sum( (alp*t**(theta-1) + alp*theta*t**(theta-1)*np.log(t))/(alp*theta*t**(theta-1) + beta*gamma*t**(gamma-1)) ) - alp*sum(t**theta*np.log(t))
        delgamma = sum( (beta*t**(gamma-1) + beta*gamma*t**(gamma-1)*np.log(t))/(alp*theta*t**(theta-1) + beta*gamma*t**(gamma-1)) ) - beta*sum(t**gamma*np.log(t))
        if x != None:
            delalp -= sum(x**theta)
            delbeta -= sum(x**gamma)
            deltheta -= alp*sum(x**theta*np.log(x))
            delgamma -= beta*sum(x**gamma*np.log(x))
        return np.array([delalp,deltheta,delbeta,delgamma])

    def numerical_grad(self,t,x=None,alp=0.5,theta=0.3,beta=0.2,gamma=0.1,eps = 1e-5):
        delalp = (self.loglik(t,x,alp+eps,theta,beta,gamma)-self.loglik(t,x,alp-eps,theta,beta,gamma))/2/eps
        deltheta = (self.loglik(t,x,alp,theta+eps,beta,gamma)-self.loglik(t,x,alp,theta-eps,beta,gamma))/2/eps
        delbeta = (self.loglik(t,x,alp,theta,beta+eps,gamma)-self.loglik(t,x,alp,theta,beta-eps,gamma))/2/eps
        delgamma = (self.loglik(t,x,alp,theta,beta,gamma+eps)-self.loglik(t,x,alp,theta,beta,gamma-eps))/2/eps
        return np.array([delalp, deltheta, delbeta, delgamma])

    def gradient_descent(self, numIter=7001, params = np.array([.5,.3,.2,0.1])):
        for i in xrange(numIter):
            if abs(params[1] - params[3]) < 1e-4:
                print "Resetting.."
                params *= np.random.uniform(size=4)*2.0
            lik = self.loglik(self.train_org,self.train_inorg,params[0],params[1],params[2],params[3])
            directn = self.grad(self.train_org,self.train_inorg,params[0],params[1],params[2],params[3])
            if sum(abs(directn) < 1e-3) == 4:
                [self.alp,self.theta,self.beta,self.gamma] = params
                return params
            elif max(directn) > 1e2:
                params = np.random.uniform(size=4)
            params2 = params
            for alp1 in [1e-8,1e-7,1e-5,1e-3,1e-2,.1,1.0,2.0]:
                params1 = params + alp1 * directn
                if(min(params1) > 0):
                    lik1 = self.loglik(self.train_org,self.train_inorg,params1[0],params1[1],params1[2],params[3])
                    if(lik1 > lik and np.isfinite(lik1)):
                        lik = lik1
                        params2 = params1
                    #else:
                    #    params2 = params + 1e-8*np.random.uniform(size=4)
            params = params2
            if i%100 == 0: #Take a Newton step.
                step = np.linalg.solve(self.hessian(self.train_org,self.train_inorg,params[0],params[1],params[2],params[3]),directn)
                params = params - step
                if min(params) < 0:
                    if False:
                        params[params<0] = np.random.uniform(size=sum(params<0))
                    else:                        
                        params = params + step
            if i%100 == 0:
                print "Iteration " + str(i) + " ,objective function: " + str(lik) + " \nparams = " + str(params) + " \nGradient = " + str(directn)
                print "\n########\n"                
        [self.alp,self.theta,self.beta,self.gamma] = params
        return params

    def hessian(self,t,x,alp=0.5,theta=0.3,beta=0.2,gamma=0.1):
        delalpsq = sum(-(theta*t**(theta-1))**2/(alp*theta*t**(theta-1) + beta*gamma*t**(gamma-1))**2)
        delbetasq = sum(-(gamma*t**(gamma-1))**2/(alp*theta*t**(theta-1) + beta*gamma*t**(gamma-1))**2)
        term1 = sum( (2*alp*beta*gamma*np.log(t)*t**(theta-1)*t**(gamma-1) + alp*beta*theta*gamma*np.log(t)**2*t**(theta-1)*t**(gamma-1) )/(alp*theta*t**(theta-1) + beta*gamma*t**(gamma-1))**2 )
        delthetasq = term1 - sum( (alp*t**(theta-1))**2 / (alp*theta*t**(theta-1) + beta*gamma*t**(gamma-1))**2 )
        delthetasq -= alp*(sum(t**theta*np.log(t)**2) + sum(x**theta*np.log(x)**2))
        term2 = sum( (2*alp*beta*theta*np.log(t)*t**(theta-1)*t**(gamma-1) + alp*beta*theta*gamma*np.log(t)**2*t**(theta-1)*t**(gamma-1) )/(alp*theta*t**(theta-1) + beta*gamma*t**(gamma-1))**2 )
        delgammasq = term2 - sum( (beta*t**(gamma-1))**2/(alp*theta*t**(theta-1) + beta*gamma*t**(gamma-1))**2 )
        delgammasq -= beta*(sum(t**gamma*np.log(t)**2) + sum(x**gamma*np.log(x)**2))
        delalpgamma = sum( (-beta*theta*t**(theta-1)*t**(gamma-1) -beta*theta*gamma*np.log(t)*t**(theta-1)*t**(gamma-1))/(alp*theta*t**(theta-1) + beta*gamma*t**(gamma-1))**2 )
        delbetatheta = sum( (-alp*gamma*t**(theta-1)*t**(gamma-1) -alp*theta*gamma*np.log(t)*t**(theta-1)*t**(gamma-1))/(alp*theta*t**(theta-1) + beta*gamma*t**(gamma-1))**2 )
        delalpbeta = sum(-theta*gamma*t**(theta-1)*t**(gamma-1)/(alp*theta*t**(theta-1) + beta*gamma*t**(gamma-1))**2)
        delthetagamma = sum(-(alp*t**(theta-1) + alp*theta*np.log(t)*t**(theta-1))*(beta*t**(gamma-1) +beta*gamma*np.log(t)*t**(gamma-1))/(alp*theta*t**(theta-1) + beta*gamma*t**(gamma-1))**2 )
        delalptheta = sum( (beta*gamma*t**(theta-1)*t**(gamma-1) + beta*theta*gamma*np.log(t)*t**(theta-1)*t**(gamma-1))/(alp*theta*t**(theta-1) + beta*gamma*t**(gamma-1))**2 )
        delalptheta -= sum(t**theta*np.log(t)) + sum(x**theta*np.log(x))
        delbetagamma = sum( (alp*theta*t**(theta-1)*t**(gamma-1) + alp*theta*gamma*np.log(t)*t**(theta-1)*t**(gamma-1))/(alp*theta*t**(theta-1) + beta*gamma*t**(gamma-1))**2 )
        delbetagamma -= sum(t**gamma*np.log(t)) + sum(x**gamma*np.log(x))
        hess = np.zeros([4,4])
        hess[0,0] = delalpsq
        hess[1,1] = delthetasq
        hess[2,2] = delbetasq
        hess[3,3] = delgammasq
        hess[0,1] = hess[1,0] = delalptheta
        hess[0,2] = hess[2,0] = delalpbeta
        hess[0,3] = hess[3,0] = delalpgamma
        hess[1,2] = hess[2,1] = delbetatheta
        hess[1,3] = hess[3,1] = delthetagamma
        hess[2,3] = hess[3,2] = delbetagamma
        return hess

    def numerical_hessian(self,t,x,alp=0.5,theta=0.3,beta=0.2,gamma=0.1):
        eps = 1e-4
        delalpsq = (self.loglik(t,x,alp+2*eps,theta,beta,gamma) + self.loglik(t,x,alp-2*eps,theta,beta,gamma) - 2*self.loglik(t,x,alp,theta,beta,gamma))/4/eps/eps
        delthetasq = (self.loglik(t,x,alp,theta+2*eps,beta,gamma) + self.loglik(t,x,alp,theta-2*eps,beta,gamma) - 2*self.loglik(t,x,alp,theta,beta,gamma))/4/eps/eps
        delbetasq = (self.loglik(t,x,alp,theta,beta+2*eps,gamma) + self.loglik(t,x,alp,theta,beta-2*eps,gamma) - 2*self.loglik(t,x,alp,theta,beta,gamma))/4/eps/eps
        delgammasq = (self.loglik(t,x,alp,theta,beta,gamma+2*eps) + self.loglik(t,x,alp,theta,beta,gamma-2*eps) - 2*self.loglik(t,x,alp,theta,beta,gamma))/4/eps/eps
        delalptheta = (self.loglik(t,x,alp+eps,theta+eps,beta,gamma) + self.loglik(t,x,alp-eps,theta-eps,beta,gamma) - self.loglik(t,x,alp+eps,theta-eps,beta,gamma) - self.loglik(t,x,alp-eps,theta+eps,beta,gamma))/4/eps/eps
        delalpbeta = (self.loglik(t,x,alp+eps,theta,beta+eps,gamma) + self.loglik(t,x,alp-eps,theta,beta-eps,gamma) - self.loglik(t,x,alp+eps,theta,beta-eps,gamma) - self.loglik(t,x,alp-eps,theta,beta+eps,gamma))/4/eps/eps
        delalpgamma = (self.loglik(t,x,alp+eps,theta,beta,gamma+eps) + self.loglik(t,x,alp-eps,theta,beta,gamma-eps) - self.loglik(t,x,alp+eps,theta,beta,gamma-eps) - self.loglik(t,x,alp-eps,theta,beta,gamma+eps))/4/eps/eps
        delthetabeta = (self.loglik(t,x,alp,theta+eps,beta+eps,gamma) + self.loglik(t,x,alp,theta-eps,beta-eps,gamma) - self.loglik(t,x,alp,theta+eps,beta-eps,gamma) - self.loglik(t,x,alp,theta-eps,beta+eps,gamma))/4/eps/eps
        delthetagamma = (self.loglik(t,x,alp,theta+eps,beta,gamma+eps) + self.loglik(t,x,alp,theta-eps,beta,gamma-eps) - self.loglik(t,x,alp,theta+eps,beta,gamma-eps) - self.loglik(t,x,alp,theta-eps,beta,gamma+eps))/4/eps/eps
        delbetagamma = (self.loglik(t,x,alp,theta,beta+eps,gamma+eps) + self.loglik(t,x,alp,theta,beta-eps,gamma-eps) - self.loglik(t,x,alp,theta,beta+eps,gamma-eps) - self.loglik(t,x,alp,theta,beta-eps,gamma+eps))/4/eps/eps
        hess = np.zeros([4,4])
        hess[0,0] = delalpsq
        hess[1,1] = delthetasq
        hess[2,2] = delbetasq
        hess[3,3] = delgammasq
        hess[0,1] = hess[1,0] = delalptheta
        hess[0,2] = hess[2,0] = delalpbeta
        hess[0,3] = hess[3,0] = delalpgamma
        hess[1,2] = hess[2,1] = delthetabeta
        hess[1,3] = hess[3,1] = delthetagamma
        hess[2,3] = hess[3,2] = delbetagamma
        return hess

    def newtonRh(self, numIter=101, params = np.array([ .5,.3,.2,.1])):
        for i in xrange(numIter):
            directn = self.grad(self.train_org,self.train_inorg,params[0],params[1],params[2],params[3])
            lik = self.loglik(self.train_org,self.train_inorg,params[0],params[1],params[2],params[3])
            step = np.linalg.solve(self.hessian(self.train_org,self.train_inorg,params[0],params[1],params[2],params[3]),directn)
            params = params - step
            if min(params) < 0:
                print "Drastic measures"
                params = params + step
                for alp1 in [1e-8,1e-7,1e-5,1e-3,1e-2,.1,.5,1.0]:
                    params1 = params + alp1 * directn
                    if(max(params1) > 0):
                        lik1 = self.loglik(self.train_org,self.train_inorg, params1[0], params1[1], params1[2], params1[3])
                        if(lik1 > lik and np.isfinite(lik1)):
                            lik = lik1
                            params2 = params1
                            scale = alp1
                params = params2
            if i%10 == 0:
                print "Iteration " + str(i) + " ,objective function: " + str(lik) + " \nparams = " + str(params) + " \nGradient = " + str(directn) + "\n##\n\n"
        [self.alp,self.theta,self.beta,self.gamma] = params
        return params

# Optimal likelihood for test dataset:
# -78962.2332699

