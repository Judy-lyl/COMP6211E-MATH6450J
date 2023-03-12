# python3 solution
#

import matplotlib.pyplot as plt
import numpy as np


class GenData:
    """
    This class generates random regression data

    Attributes:
            x: n x d matrix of input data
            y: n dimensional vector of response
    """
    def __init__(self):
        """
        The constructor, with fixed data generation
        """
        np.random.seed(12345)
        d=1000
        ntrn=100
        sigma=1
        temp=np.ones((d,1))/np.linspace(1,500,d).reshape((d,1))
        wtrue=np.sqrt(temp)
        xtrn = np.random.randn(ntrn,d).dot(np.diagflat(temp))
        ytrn = xtrn.dot(wtrue) + sigma* np.random.randn(ntrn,1)
        self.x=xtrn
        self.y=ytrn
    

class RidgeObj:
    """
    This class provides an interface to the ridge regression objective function
    f(w) = 0.5 [\| x* w - y\|_2^2 + lam * w *w]
    we normalize x and y by dividing sqrt{n}

    Attributes:
        x: n x d matrix of normalized input data
        y: n dimensional vector of normalized response
        lam: regularization parameter
        wstar: closed form solution
        fstar: optimal objective function value at wstar
    """

    def __init__(self,data,lam):
        """
        The constructor
        :param data: generated data
        :param lam: regularization parameter
        """
        n=np.size(data.y)
        self.x=data.x/np.sqrt(n)
        self.y=data.y/np.sqrt(n)
        self.lam=lam
        self.__solve__()

    def L(self):
        """
        This function compute the smoothness parameter of the objective
        :return: the smoothness parameter
        """
        u,s,vh=np.linalg.svd(self.x)
        s=np.amax(s)
        return s*s+self.lam

    def __solve__(self):
        """
        This function computes the closed form solution of the ridge regression problem
        It then sets self.wstar and self.fstar and self.hessian_inv
        :return:
        """
        d=np.size(self.x,1)
        self.hessian_inv = np.linalg.inv(self.x.transpose().dot(self.x)+self.lam*np.eye(d))
        self.wstar= self.hessian_inv.dot(self.x.transpose().dot(self.y))
        self.fstar=self.obj(self.wstar)
        return

    def obj(self,w):
        """
        This function computes the objective function value
        :param w: parameter at which to compute f(w)
        :return: f(w)
        """
        res=self.x.dot(w)-self.y
        return 0.5*(res.transpose().dot(res) +self.lam*(w.transpose().dot(w))).item()

    def grad(self,w):
        """
        This function computes the gradient of the objective function
        :param w: parameter at which to compute gradient
        :return: gradient of f(w)
        """
        res=self.x.dot(w)-self.y
        return self.x.transpose().dot(res)+self.lam*w



def PGD(ridge, w0, eta, t, invH = None):
    """
    This function performs (preconditioned) gradient descent for t iterations
    :param ridge: ridge objective function class
    :param w0: initial parameter
    :param eta: learning rate
    :param t: number of iterations
    :function invH: preconditioned matrix function with input w; `None` (default) denotes no preconditioning
    :return: t+1 function values evaluated at the intermediate solutions
    """
    w=w0
    fv=np.zeros(t+1)
    fv[0]=ridge.obj(w)
    if invH is None:
        grad = lambda w: ridge.grad(w)
    else:
        grad = lambda w: np.matmul(invH(w),ridge.grad(w))
    for ti in range(t):
        w= w - eta*grad(w)
        fv[ti+1]=ridge.obj(w)
    return fv


def main():
    # generate data
    data=GenData()

    # gradient descent
    lam_arr=[1e-4,1e-2,1,1e1]
    for lam in lam_arr:
        ridge = RidgeObj(data,lam)
        # trying different learning rate at 0.1/L L 2/L
        eta_lip_list = np.array([0.1,1,2])

        plt.xlabel('iterations')
        plt.ylabel('primal-suboptimality')
        
        for eta_lip in eta_lip_list:
            eta = eta_lip/ridge.L()
            w0 = np.zeros((np.size(ridge.wstar,0),1))
            t = 100
            # perform gradient descent and return function values, compute primal suboptimality
            subopt = PGD(ridge,w0,eta,t)-ridge.fstar
            leg = '$\eta$={}/L'.format(eta_lip)
            plt.plot(np.arange(t+1),subopt,label=leg)
        plt.legend()
        plt.yscale('log')
        plt.title('$f_*=%.4f$, $\lambda=%.4f$, $L=%.4f$'%(ridge.fstar,lam,ridge.L()))
        plt.subplots_adjust(left=0.15)
        filename = 'plot-lam={}.pdf'.format(lam,eta)
        
        plt.savefig(filename)
        plt.close()
    
    # preconditioned gradient descent
    lam = 1e-2
    w0 = np.zeros((np.size(ridge.wstar,0),1))
    ridge = RidgeObj(data,lam)

    t = 1
    eta = 1

    subopt=PGD(ridge,w0,eta,t,lambda x: ridge.hessian_inv)-ridge.fstar
    assert subopt[-1]<1e-15
    print("PGD converges in %d step(s) with eta %f"%(t,eta))
        
if __name__ == "__main__":
    main()

    
 
