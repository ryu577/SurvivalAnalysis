import numpy as np
import matplotlib.pyplot as plt

class Base():

    def Ex_x_le_y(self,xs = np.arange(1,100000)*0.01):
        """
        Numerically integrates the PDF and obtains the expected value of x conditional on x less than y.
        """
        vals = []
        # \int_0^t x f_x dx
        for i in xs:
            vals.append((i+0.005)*self.pdf(i+0.005)*0.01)
        return np.cumsum(vals)

    def expected_downtime(self,Y,xs=np.arange(1,100000)*0.01,lmb=0,reg='log'):
        """
        Combines the expected downtime when the recovery happens before and after the wait threshold.
        """
        highterms = self.survival(xs)*(xs+Y)
        lowterms = self.Ex_x_le_y(xs)
        et = lowterms + highterms
        if reg == 'log':
            et+= lmb*np.log(xs)
        elif reg == 'sqrt':
            et+= lmb*xs**.5
        elif reg == 'sqr':
            et+= lmb*xs**2
        return et

    def prob_TgrTau(self,xs=np.arange(1,100000)*0.01,lmb=0.2,t0=900.0,Y=480.0):
        return lmb*((xs>t0)*(self.survival(t0)-self.survival(xs)) + (xs> (t0-Y))*self.survival(xs))

    def plt_downtime(self,xs=np.arange(1,100000)*0.01,lmb=0,alp=1,lmb_prob=0,t0=900.0,Y=480.0,reg='log',col='b'):
        ys = self.expected_downtime(480.0,xs=xs,lmb=lmb,reg=reg)
        ys_probs = self.prob_TgrTau(xs,lmb_prob,t0,Y)
        plt.plot(xs,(ys+ys_probs),alpha=alp,color=col)
        return (ys+ys_probs)


