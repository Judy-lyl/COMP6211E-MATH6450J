#!/usr/local/bin/python3
#

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

class BinaryClassificationData:
    """
    This class loads binary classification data from file

    Attributes:
            x: n x d matrix of input data
            y: n dimensional vector of [+-1] response
    """

    def __init__(self, file):
        """
        load binary classification data from file
        :param file: data file (with +1 and -1 binary labels)
               format: label (+1/-1) followed by features
               it assumes the features in [0,255] and scales it to [0,1]
        """
        data_txt = np.loadtxt(file, delimiter=",")
        self.y = np.asfarray(data_txt[:, :1])
        self.x = np.asfarray(data_txt[:, 1:]) * (0.99 / 255) + 0.01


class BinaryLinearClassifier:
    """
    This class is linear classifier
    Attributes:
        w: linear weights
    """

    def __init__(self, w):
        """
        :param w: linear weights
        """
        self.w= w.reshape((-1,1))

    def classify(self,x):
        """
        classify data x
        :param x: data matrix to be classified
        :return: class labels
        """
        yp=x.dot(self.w)
        return (yp>=0)*2-1

    def test_error_rate(self,data):
        """
        compute test error rate on data
        :param data: data to be evaluated
        :return: test error rate
        """
        yp=self.classify(data.x)
        return np.sum(data.y*yp<=0)/np.size(data.y)

    def nnz(self):
        """
        sparsity
        :return: number of nonzero weights
        """
        return sum(np.abs(self.w)>1e-10)


class LogisticObj:
    """
    This class provides an interface to the L1-L2 regularized logistic regression objective function
    phi(w) = f(w) + g(w)
    f(w) = log(1+exp(-w*x*y)
    g(w) = 0.5* lam * w *w + mu * ||w||_1

    Attributes:
        x: n x d matrix of normalized input data
        y: n dimensional binary classification responses {-1 +1} values
        lam: L2-regularization parameter
        mu: L1-regularization parameter
        wstar: approximate solution
    """

    def __init__(self, data, lam, mu):
        """
        The constructor
        :param data: generated data
        :param lam: regularization parameter
        """
        n = np.size(data.y)
        self.x = data.x
        self.y = data.y
        self.lam = lam
        self.mu = mu
        self.__solve__()

    def L(self):
        """
        This function compute the smoothness parameter of the objective
        :return: the smoothness parameter
        """
        n = np.size(self.x, 0)
        u, s, vh = np.linalg.svd(self.x)
        s = np.amax(s)
        return (0.25 / n) * s * s

    def __solve__(self):
        """
        This function computes an approximate solution of the objective
        It then sets self.wstar
        :return:
        """
        d = np.size(self.x, 1)
        w0 = np.zeros((d, 1))
        t = 10000
        alpha = 0.5 
        beta = 0.9
        ww = ProxACCL.solve(self, w0, alpha,beta, t)
        self.wstar = ww[:, t].reshape(-1, 1)
        print('approximate optimal solution: norm of final prox-gradient={:.2g}'.format(np.linalg.norm(self.grad_prox(alpha, self.wstar), 2)))
        return

    def obj(self, w):
        """
        This function computes the objective function value
        :param w: parameter at which to compute f(w)+g(w)
        :return: phi(w)=f(w)+g(w)
        """
        obj_g= 0.5 * self.lam * (w.transpose().dot(w)) + self.mu * np.linalg.norm(w, 1)
        return self.obj_f(w) + obj_g

    def grad(self, w):
        """
        This function computes the gradient of the objective f(w)+g(w)
        :param w: parameter at which to compute gradient
        :return: gradient of f(w) + g(w)
        """
        grad_g=self.lam * w + self.mu * np.sign(w)
        return self.grad_f(w) + grad_g

    def grad_prox(self, alpha, w):
        """
        This function computes prox gradient of the objective f(w) + g(w)
        :param alpha: learning rate
        :param w: parameter at which to compute proximal gradient
        :return: prox_grad(w) = (w- prox(w- alpha* nabla f(w)))/alpha
        """
        wt = w - alpha * self.grad_f(w)
        ww = self.prox_map(alpha, wt)
        return (w - ww) / alpha

    def obj_f(self, w):
        """
        This function computes the objective function value of f(x)
        :param w: parameter at which to compute f(w)
        :return: f(w)
        """
        wp = w.reshape(np.size(w), 1)
        loss = np.log(1 + np.exp(-self.x.dot(wp) * self.y))
        return np.mean(loss)

    def grad_f(self, w):
        """
        This function computes the gradient of the objective function
        :param w: parameter at which to compute gradient
        :return: gradient of f(w)
        """
        wp = w.reshape(np.size(w), 1)
        dloss = -1 / (1 + np.exp(self.x.dot(wp) * self.y))
        return self.x.transpose().dot(dloss * self.y) / np.size(dloss)

    def prox_map(self, eta, w):
        """
        compute the proximal mapping \arg\min_u [ (0.5/eta) * ||u-w||_2^2 + g(w) ]
        :param eta: learning rate
        :param w: parameter to compute proximal mapping
        :return: proximal_map
        """
        wp = w/(1+self.lam*eta)
        etap=eta/(1+self.lam*eta)
        u = np.maximum(0, np.abs(wp) - etap * self.mu) * np.sign(wp)
        return u


class RDA:
    """
    Implementing RDA of Lecture 13
    """
    @staticmethod
    def solve(phi,x0,eta,t):
        """
        solve min_x f(x) + g(x) using RDA
        :param phi(x)=f(x)+g(x): objective function to be minimized (require phi.grad_f() and phi.prox_map())
        :param x0: initial point
        :param eta: learning rate
        :param t: number of iterations
        :return: the t+1 iterates found by the method from x_0 to x_t
        """
        xp = x0
        d = np.size(x0)
        result = np.zeros((d, t + 1))
        result[:, 0] = x0.transpose()
        xt=x0
        etat=0
        for ti in range(t):
            etat=etat+eta
            xt = xt - eta * phi.grad_f(xp)
            x = phi.prox_map(etat, xt)
            xp = x
            result[:, ti + 1] = x.transpose()
        return result


class ProxGD:
    """
    Implementing Proximal Gradient Descent Algorithm
    """

    @staticmethod
    def solve(phi,x0,eta,t):
        """
        solve min_x f(x)+g(x) using proximal gradient descent
        :param phi(x)=f(x)+g(x): objective function to be minimized (require phi.grad_f() and phi.prox_map())
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
            xt=xp-eta*phi.grad_f(xp)
            x= phi.prox_map(eta,xt)
            xp=x
            result[:,ti+1]=x.transpose()
        return result

class ProxACCL:
    """
    Implement Nesterov's Accelerated Proximal Gradient Algorithm
    """
    @staticmethod
    def solve(phi,x0,alpha,beta,t):
        """
        solve min_x phi(x) := f(x)+g(x) using Nesterov's Acceleration
        :param phi: objective function to be minimized (require phi.grad_f() and phi.prox_map())
        :param x0: initial point
        :param alpha:  learning rate
        :param beta: momentum parameter
        :param t: number of iterations
        :return: the t+1 iterates found by ACCL from x_0 to x_t
        """
        xp=x0
        xpp=xp
        d=np.size(x0)
        result=np.zeros((d,t+1))
        result[:,0]=x0.transpose()
        for ti in range(t):
            y=xp+beta*(xp-xpp)
            xt=y-alpha*phi.grad_f(y)
            x=phi.prox_map(alpha,xt)
            xpp=xp
            xp=x
            result[:,ti+1]=x.transpose()
        return result


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

    @staticmethod
    def solve_AG(f,x0,eta0,t):
        """
        solve min_x f(x) using gradient descent
        :param f: objective function to be minimized (require f.grad())
        :param x0: initial point
        :param eta0: initial learning rate
        :param t: number of iterations
        :return: the t+1 iterates found by GD from x_0 to x_t
        """
        tau=0.8
        c=0.5
        xp=x0
        d=np.size(x0)
        result=np.zeros((d,t+1))
        result[:,0]=x0.transpose()
        fp=f.obj(xp)
        eta=eta0
        for ti in range(t):
            g=f.grad(xp)
            g2=np.linalg.norm(g,2)
            x=xp-eta*g
            fc=f.obj(x)
            if (g2>1e-8):
                etap=(fp-fc)/g2**2
                while (etap<=c*eta and eta >1e-4*eta0):
                    eta=eta*tau
                    x=xp-eta*g
                    fc=f.obj(x)
                    etap=(fp-fc)/g2**2
                if (etap>=c*eta/tau):
                    eta=eta/np.sqrt(tau)
            fp=fc
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


class MyFigure:

    def __init__(self,lam,mu,filename):
        plt.rcParams.update({'font.size': 12})
        fig, axs = plt.subplots(2, 2,figsize=(12,8))
        fig.suptitle(r'iterations ($\lambda$={:.2g} $\mu$={:.2g})'.format(lam, mu))

        axs[0, 0].set_ylabel('primal-suboptimality')
        axs[0, 1].set_ylabel('weight sparsity')
        axs[1, 0].set_ylabel('training error rate')
        axs[1, 1].set_ylabel('test error rate')
        self.fig=fig
        self.axs=axs
        self.filename=filename
        self.marker0=10

    def finish(self):
        self.axs[0, 0].legend()
        self.axs[0, 1].legend()
        self.axs[1, 0].legend()
        self.axs[1, 1].legend()
        self.axs[0, 0].set_yscale('log')
        self.axs[1, 0].set_yscale('log')
        self.axs[1, 1].set_yscale('log')
        self.fig.savefig(self.filename + '.pdf')
        plt.close(self.fig)

    def plot(self,phi, result, col, lab,train_data,test_data):
        """
        plot the convergence result for a method
        :param phi: function to be evaluated
        :param result:  iterates generated by optimization algorithm from 0 to t-1
        :param col: plot color
        :param lab: plot label
        :return: none
        """

        t = np.size(result, 1)
        xx = np.arange(t)
        yy = np.zeros((t, 1))
        trnerr= np.zeros((t,1))
        tsterr = np.zeros((t, 1))
        nnz= np.zeros((t,1))
        w = np.zeros((np.size(result, 0), 1))
        phi_star=phi.obj(phi.wstar)
        for ti in range(t):
            w[:, 0] = result[:, ti]
            yy[ti] = np.maximum(phi.obj(w) - phi_star, 1e-16)
            lc=BinaryLinearClassifier(w)
            trnerr[ti]=lc.test_error_rate(train_data)
            tsterr[ti] = lc.test_error_rate(test_data)
            nnz[ti]=lc.nnz()
        self.axs[0,1].plot(xx, nnz, col, markevery=(self.marker0,100), linestyle='dashed',  label=lab)
        self.axs[0,0].plot(xx, yy, col, markevery=(self.marker0,100), linestyle='dashed',  label=lab)
        self.axs[1,0].plot(xx, trnerr, col, markevery=(self.marker0,100), linestyle='dashed',  label=lab)
        self.axs[1,1].plot(xx, tsterr, col, markevery=(self.marker0,100), linestyle='dashed',   label=lab)
        self.marker0+=25




class Visualize:

    def __init__(self,lam,mu,filename,algorithms=["" for i in range(4)]):
        plt.rcParams.update({'font.size': 12})
        fig, axs = plt.subplots(2, 2,figsize=(12,8))
        fig.suptitle(r'weight (absolute value) visualization ($\lambda$={:.2g} $\mu$={:.2g})'.format(lam, mu))

        axs[0, 0].set_ylabel(algorithms[0])
        axs[0, 1].set_ylabel(algorithms[1])
        axs[1, 0].set_ylabel(algorithms[2])
        axs[1, 1].set_ylabel(algorithms[3])
        self.fig=fig
        self.axs=axs
        self.filename=filename
        self.marker0=10

    def finish(self):
        self.fig.savefig(self.filename + '.pdf')
        plt.close(self.fig)

    def plot(self, result1, result2, result3, result4):
        a1 = self.axs[0,0].imshow((np.abs(result1[:,-1]+1e-20).reshape(28,28)),cmap='gray',norm=LogNorm(vmin=1e-10, vmax=10))
        a2 = self.axs[0,1].imshow((np.abs(result2[:,-1]+1e-20).reshape(28,28)),cmap='gray',norm=LogNorm(vmin=1e-10, vmax=10))
        a3 = self.axs[1,0].imshow((np.abs(result3[:,-1]+1e-20).reshape(28,28)),cmap='gray',norm=LogNorm(vmin=1e-10, vmax=10))
        a4 = self.axs[1,1].imshow((np.abs(result4[:,-1]+1e-20).reshape(28,28)),cmap='gray',norm=LogNorm(vmin=1e-10, vmax=10))
        self.fig.colorbar(a1,ax = self.axs[0,0])
        self.fig.colorbar(a2,ax = self.axs[0,1])
        self.fig.colorbar(a3,ax = self.axs[1,0])
        self.fig.colorbar(a4,ax = self.axs[1,1])
        self.marker0+=25


def do_experiment(filename,lam,mu,train_data,test_data):

    print("solving L1-L2 regularized logistic regression with lambda={:.2g} mu={:.2g}".format(lam,mu))

    phi=LogisticObj(train_data,lam,mu)

    w0 = np.zeros((np.size(phi.wstar, 0), 1))

    # compare ProxGD proxACCL to GD ACCL
    #
    t=200
    eta = 1.0
    beta = 0.9
    myfig=MyFigure(lam,mu,filename+'-prox')
    myfig2=Visualize(lam,mu,filename+'-prox-visualize',["ProxGD","ProxACCL","GD","ACCL"])

    result1 = ProxGD.solve(phi, w0, eta, t)

    myfig.plot(phi, result1, 'ko', 'ProxGD',train_data,test_data)

    result2=ProxACCL.solve(phi,w0,eta,beta,t)
    myfig.plot(phi,result2,'b+','ProxACCL',train_data,test_data)

    result3 = GD.solve(phi, w0, eta, t)
    myfig.plot(phi, result3, 'rx', 'GD',train_data,test_data)

    result4=ACCL.solve(phi,w0,eta,beta,t)
    myfig.plot(phi,result4,'g','ACCL',train_data,test_data)
    myfig2.plot(result1,result2,result3,result4)
    myfig.finish()
    myfig2.finish()
    # compare GD to ProxGD to RDA 
    #
    
    t=200
    eta=1.0
    myfig = MyFigure(lam, mu,filename+'-dual')
    myfig2=Visualize(lam,mu,filename+'-dual-visualize',["GD","ProxGD","RDA","ACCL"])


    result1 = GD.solve(phi, w0, eta, t)
    myfig.plot(phi, result1, 'ko', 'GD', train_data, test_data)

    result2 = ProxGD.solve(phi, w0, eta, t)
    myfig.plot(phi, result2, 'b+', 'ProxGD', train_data, test_data)

    result3 = RDA.solve(phi, w0, eta, t)
    myfig.plot(phi, result3, 'rx', 'RDA', train_data, test_data)
    myfig2.plot(result1,result2,result3,result4)
    myfig.finish()
    myfig2.finish()
    

def main():
    train_data=BinaryClassificationData("mnist/mnist_train_binary.csv")
    test_data=BinaryClassificationData("mnist/mnist_test_binary.csv")
    lam=1e-4
    mu=1e-2
    filename="fig-1"
    do_experiment(filename,lam,mu,train_data,test_data)

    
    lam=1e-4
    mu=1e-4
    filename="fig-2"
    do_experiment(filename,lam,mu,train_data,test_data)
    
if __name__ == "__main__":
    main()
