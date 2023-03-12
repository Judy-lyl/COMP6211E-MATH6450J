# python3 solution
#

import matplotlib.pyplot as plt
import numpy as np


class HingeData:
    """
    This class generates random binary classification data

    Attributes:
            x: n x d matrix of input data
            y: n dimensional vector of [+-1] response
    """

    def __init__(self):
        """
        The constructor, with fixed data generation
        """
        np.random.seed(12345)
        d = 200
        ntrn = 50
        temp = np.ones((d, 1)) / np.linspace(1, 500, d).reshape((d, 1))
        wtrue = np.sqrt(temp)
        xtrn = np.random.rand(ntrn, d).dot(np.diagflat(temp))
        ptrn = 1 / (1 + np.exp(-xtrn.dot(wtrue)))
        ytrn = (np.random.rand(ntrn, 1) < ptrn) * 2 - 1
        self.x = xtrn
        self.y = ytrn
        self.wtrue = wtrue




class HingeObj:
    """
    This class provides an interface to the Smoothed Hinge-Loss (SVM) objective function
    f(w) = smoothed-hinge(1-w*x*y) + 0.5* lam * w *w

           smoothed-hinge(z) =   min_u [ (u)_+ + (0.5/gamma) (u-z)^2 ]

    Attributes:
        x: n x d matrix of normalized input data
        y: n dimensional binary classification responses {-1 +1} values
        lam: regularization parameter
        gamma: smoothing parameter
        wstar: approximate solution
        fstar: optimal objective function value at wstar
    """

    def __init__(self,data,lam,gamma):
        """
        The constructor
        :param data: generated data
        :param lam: regularization parameter
        """
        n=np.size(data.y)
        self.x=data.x
        self.y=data.y
        self.lam=lam
        self.data = data
        self.gamma=gamma
        self.__solve__()


    def __solve__(self):
        """
        This function computes an approximate solution using acceleration
        It then sets self.wstar and self.fstar
        :return:
        """
        d=np.size(self.x,1)
        w0=np.zeros((d,1))
        t=100000
        n = np.size(self.x, 0)
        u, s, vh = np.linalg.svd(self.x)
        s = np.amax(s)
        eta=0.2/((1/n)*(1/np.amax([1e-5,self.gamma]))*s*s+self.lam)

        ww = ACCL.solve_general(self, w0, eta, self.lam, t)
        self.wstar= ww[:,t].transpose()
        self.fstar=self.obj(self.wstar)
        print('fstar = {}'.format(self.fstar.item()))
        print ('norm of final gradient={:.2g}'.format(np.linalg.norm(self.grad(self.wstar),2)))

        cosine_sim = (np.dot(self.wstar,self.data.wtrue)/np.linalg.norm(self.wstar)/np.linalg.norm(self.data.wtrue)).item()
        print('cosine similarity between wtrue and wstar is {:.5g}'.format(cosine_sim))
        return

    def obj(self,w):
        """
        This function computes the objective function value
        :param w: parameter at which to compute f(w)
        :return: f(w)
        """
        wp=w.reshape(np.size(w),1)
        z=1-self.x.dot(wp)*self.y
        if (self.gamma>0):
            loss= (z>self.gamma)*(z-self.gamma/2)+(z<=self.gamma)*(z>0)*(0.5/self.gamma)*z**2
        else:
            loss=(z>0)*(z)
        return np.mean(loss) + 0.5*self.lam*(wp.transpose().dot(wp))

    def grad(self,w):
        """
        This function computes the gradient of the objective function
        :param w: parameter at which to compute gradient
        :return: gradient of f(w)
        """
        wp=w.reshape(np.size(w),1)
        z=1-self.x.dot(wp)*self.y
        if (self.gamma>0):
            dloss=-1*(z>self.gamma)+(z<=self.gamma)*(z>0)*z*(-1/self.gamma)
        else:
            dloss=-1*(z>0)
        return self.x.transpose().dot(dloss*self.y)/np.size(dloss)+self.lam*wp

class GD:
    """
    Implementing Gradient Descent Algorithm
    """

    @staticmethod
    def solve(f,x0,eta,t):
        """
        solve min_x f(x) using gradient descent
        :param f: objective function to be minimized (require f.grad())
        :param x0: initial point
        :param eta: learning rate
        :param t: number of iterations
        :return: the t+1 iterates found by GD from x_0 to x_t
        """
        xp=x0
        d=np.size(x0)
        result=np.zeros((d,t+1))
        result[:,0]=x0.transpose()
        for ti in range(t):
            x=xp-eta*f.grad(xp)
            xp=x
            result[:,ti+1]=x.transpose()
        return result


class HB :
    """
    Implement Heavy Ball Algorithm
    """

    @staticmethod
    def solve(f,x0,alpha,beta,t):
        """
        solve min_x f(x) using Heavy Ball
        :param f: objective function to be minimized (require f.grad())
        :param x0: initial point
        :param alpha: learning rate
        :param beta: momentum parameter
        :param t: number of iterations
        :return: the t+1 iterates found by HB from x_0 to x_t
        """
        xp=x0
        xpp=x0
        d=np.size(x0)
        result=np.zeros((d,t+1))
        result[:,0]=x0.transpose()
        for ti in range(t):
            y=xp+beta*(xp-xpp)
            x=y-alpha*f.grad(xp)
            xpp=xp
            xp=x
            result[:,ti+1]=x.transpose()
        return result
    
    @staticmethod
    def solve_adaptive(f,x0,alpha,t):
        """
        solve min_x f(x) using Heavy Ball (with adaptive beta)
        :param f: objective function to be minimized (require f.grad())
        :param x0: initial point
        :param alpha: learning rate
        :param t: number of iterations
        :return: the t+1 iterates found by HB from x_0 to x_t
        """
        xp=x0
        xpp=x0
        d=np.size(x0)
        result=np.zeros((d,t+1))
        result[:,0]=x0.transpose()
        gp=np.linalg.norm(f.grad(xp),2)
        lrate=0
        for ti in range(t):
            beta=min(1,np.exp(lrate))
            y=xp+beta*(xp-xpp)
            p=f.grad(xp)
            g=np.linalg.norm(p,2)
            lrate=0.8*lrate+0.2*2*np.log(g/gp)
            gp=g
            x=y-alpha*p
            xpp=xp
            xp=x
            result[:,ti+1]=x.transpose()
        return result
    
class ACCL:
    """
    Implement Nesterov's Acceleration Algorithm
    """

    @staticmethod
    def solve(f,x0,alpha,beta,t):
        """
        solve min_x f(x) using Nesterov's Acceleration
        :param f: objective function to be minimized (require f.grad())
        :param x0: initial point
        :param alpha:  learning rate
        :param beta: momentum parameter
        :param t: number of iterations
        :return: the t+1 iterates found by ACCL from x_0 to x_t
        """
        xp=x0
        xpp=x0
        d=np.size(x0)
        result=np.zeros((d,t+1))
        result[:,0]=x0.transpose()
        for ti in range(t):
            y=xp+beta*(xp-xpp)
            x=y-alpha*f.grad(y)
            xpp=xp
            xp=x
            result[:,ti+1]=x.transpose()
        return result


    @staticmethod
    def solve_adaptive(f,x0,alpha,t,eps=1e-16):
        """
        solve min_x f(x) using Nesterov's Acceleration
        :param f: objective function to be minimized (require f.grad())
        :param x0: initial point
        :param alpha:  learning rate
        :param beta: momentum parameter
        :param t: number of iterations
        :param eps: stopping criterion of gradient
        :return: the t+1 iterates found by ACCL from x_0 to x_t
        """
        xp=x0
        xpp=x0
        d=np.size(x0)
        result=np.zeros((d,t+1))
        result[:,0]=x0.transpose()
        gp=np.linalg.norm(f.grad(xp),2)
        lrate=0
        for ti in range(t):
            beta=min(1, np.exp(lrate))

            y=xp+beta*(xp-xpp)
            p=f.grad(y)
            g=np.linalg.norm(p,2)
            if (g<1e-16):
                break
            lrate=0.8*lrate+0.2*2*np.log(g/gp)
            gp=g
            x=y-alpha*p
            xpp=xp
            xp=x
            result[:,ti+1]=x.transpose()
        return result
    
    @staticmethod
    def solve_general(f,x0,alpha,lam,t):
        """
        solve min_x f(x) using Nesterov's Acceleration
        :param f: objective function to be minimized (require f.grad())
        :param x0: initial point
        :param alpha:  learning rate
        :param lam: strong convexity parameter
        :param t: number of iterations
        :return: the t+1 iterates found by ACCL from x_0 to x_t
        """
        gamma=1/alpha
        theta=1
        xp=x0
        xpp=x0
        d=np.size(x0)
        result=np.zeros((d,t+1))
        result[:,0]=x0.transpose()
        for ti in range(t):
            thp=theta
            theta=((lam-gamma)*alpha+np.sqrt(((lam-gamma)*alpha)**2+4*gamma*alpha))/2
            beta=(1/theta-1)*(1/thp-1)*gamma/(1/alpha-lam)
            gamma=(1-theta)*gamma+theta*lam
            y=xp+beta*(xp-xpp)
            x=y-alpha*f.grad(y)
            xpp=xp
            xp=x
            result[:,ti+1]=x.transpose()
        return result



def plot_conv(f,result,col,lab) :
    """
    plot the convergence result for a method
    :param f: function to be evaluated (require f.obj() and f.fstar)
    :param result:  iterates generated by optimization algorithm from 0 to t-1
    :param col: plot color
    :param lab: plot label
    :return: none
    """

    t=np.size(result,1)
    xx=np.arange(t)
    yy=np.zeros((t,1))
    w=np.zeros((np.size(result,0),1))
    for ti in range(t):
        w[:,0]=result[:,ti]
        yy[ti]=np.maximum(f.obj(w)-f.fstar,1e-16)
    plt.plot(xx,yy,linestyle='dashed',color=col,label=lab)


def do_experiment(ff,lam,gamma,alpha,beta):

    t=1000

    w0=np.zeros((np.size(ff.wstar,0),1))

    plt.xlabel('iterations (gamma={:.2g} lambda={:.2g})'.format(gamma, lam))
    plt.ylabel('primal-suboptimality')

    w=w0
    result=GD.solve(ff,w,alpha,t)
    plot_conv(ff,result,'black','GD')

    w=w0
    result=ACCL.solve(ff,w,alpha,beta,t)
    plot_conv(ff,result,'green','Accl-beta={:.2g}'.format(beta))

    w = w0
    result = HB.solve(ff, w, alpha, beta, t)
    plot_conv(ff, result, 'blue', 'HB-beta={:.2g}'.format(beta))

    w=w0
    result=ACCL.solve_general(ff,w,alpha,0,t)
    plot_conv(ff,result,'red','Accl-General')
   

    plt.legend()
    plt.yscale('log')

    filename='plot-gamma={:.2g}-lambda={:.2g}-beta={:.2g}.pdf'.format(gamma,lam,beta)
    plt.subplots_adjust(left=0.15)
    plt.savefig(filename)

    plt.close()


def main():
    data=HingeData()
    # smooth and strongly convex function optimization
    print("smooth and strongly convex optimization")
    lam = 1e-2
    gamma = 1
    ff = HingeObj(data, lam, gamma)

    alpha=1
    for beta in [0.5,0.9,0.95]:
        do_experiment(ff,lam,gamma,alpha,beta)

    # nearly nonsmooth function optimization
    print("nearly nonsmooth and strongly convex optimization")
    lam = 1e-2
    gamma = 1e-3
    ff = HingeObj(data, lam, gamma)

    alpha=0.01
    for beta in [0.9,0.95,0.99]:
        do_experiment(ff,lam,gamma,alpha,beta)

    # nearly nonstrongly convex function optimization
    print("smooth and nearly nonstrongly convex optimization")
    lam = 1e-6
    gamma = 1e-1
    ff = HingeObj(data, lam, gamma)

    alpha=0.1
    for beta in [0.9,0.95,0.99]:
        do_experiment(ff,lam,gamma,alpha,beta)

    # the best choice should be lam = 1e-2 gamma = 1e-3 with Accl or HB. (the fastest way to get the best prediction)

    # For different lambda, one can find that 
    # (1) Considering the optimization process, lambda determines the coefficent of stronly convexity, which leads to faster convergence;
    # (2) Considering the generalization (prediction), lambda can regularize the model to avoid overfitting effectively before exceeding a certain threshold.




if __name__ == "__main__":
    main()
