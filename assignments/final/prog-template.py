#!/usr/local/bin/python3
#

import matplotlib.pyplot as plt
import numpy as np


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



class RegL1L2:
    """
    Implement the L1-L2 regularizer g(w) = 0.5 lam * \|w\|_2^2 + mu * \|w\|_1
    """
    def __init__(self,lam,mu):
        """
        initialization
        :param lam: L2-regularization parameter
        :param mu: L1-regularization parameter
        """
        self.lam=lam
        self.mu=mu

    def param_smoothness(self):
        """
        smoothness parameter
        :return: smoothness
        """
        return self.lam + (self.mu>0)*1e5

    def param_strongconvex(self):
        """
        strong convexity parameter
        :return: self.lam
        """
        return self.lam

    def obj(self,w):
        """
        This function computes g(w) : the objective value of the regularizer
        :param w: model parameter
        :return: g(w)
        """
        return 0.5 * self.lam * (w.transpose().dot(w)) + self.mu * np.linalg.norm(w, 1)

    def grad(self,w):
        """
        This function computes the gradient of the regularizer g(w)
        :param w: model parameter
        :return: gradient of g(w)
        """
        return self.lam * w + self.mu * np.sign(w)

    def prox_map(self, eta, w, i=-1):
        """
        compute the proximal mapping \arg\min_u [ (0.5/eta) * \|u-w\|_2^2 + g_i(w) ]
        :param eta: learning rate
        :param w: parameter to compute proximal mapping
        :param i: the i-th component of g() (-1 is all, not used in the current implementation)
        :return: proximal_map
        """
        wp = w / (1 + self.lam * eta)
        etap = eta / (1 + self.lam * eta)
        u = np.maximum(0, np.abs(wp) - etap * self.mu) * np.sign(wp)
        return u


class LossLogistic:
    """
    Implement the Logistic Loss f(w) = log(1+ exp(-w*x*y))
    """
    def __init__(self):
        pass

    def param_smoothness(self,x,y,quick=True):
        """
        This function compute the smoothness parameter of f(w,x,y)
         :param x: feature matrix
        :param y: target response
        :param quick: quick estimation
        :return: the smoothness parameter
        """
        myL=0.25
        if (quick):
            myL = myL*np.mean(np.sum(x * x, axis=1))
        else:
            u, s, vh = np.linalg.svd(x)
            s = np.amax(s)
            myL=(myL / np.size(x,0)) * s * s
        return myL

    def obj(self,w,x,y):
        """
        This function compute the loss f(w,x,y)
        :param x: feature matrix
        :param y: target response
        :param w: model parameter
        :return: f(w,x,y)
        """
        wp=w.reshape(np.size(w),1)
        loss=np.log(1+np.exp(-x.dot(wp)*y))
        return np.mean(loss)[0,0]


    def grad(self, w,x,y):
        """
        This function computes the gradient of f(w,x,y)
        :param w: parameter at which to compute gradient
        :return: gradient of f(w,x,y)
        """
        wp=w.reshape(np.size(w),1)
        dloss=-1/(1+np.exp(x.dot(wp)*y))
        return x.transpose().dot(dloss*y)/np.size(dloss)

class RegularizedLoss:
    """
    Implement regularized loss of the form:
       phi(w)= f(w) + g(w)
               loss + regularizer
    Atributes:
            data:  (data.x data.y)
            f:    loss f(w)
            g:    regularizer g(w)= sum_i g_i(w_i)
    """
    def __init__(self,data,loss,reg):
        """
        init
        :param data: training data
        :param loss: loss function
        :param reg: regularizer
        """
        self.data=data
        self.f=loss
        self.g=reg


    def set_learning_rate(self):
        """"
        set learning rate to be 1/f-smoothness
        """
        return 1.0/self.f.param_smoothness(self.data.x,self.data.y,True)

    def obj_f(self,w):
        """
        compute loss objective f(w)
        :param w: model parameter
        :return: f(w)
        """
        return self.f.obj(w,self.data.x,self.data.y)

    def obj(self,w):
        """
        compute regularized-loss objective f(w) + g(w)
        :param w: model parameter
        :return: f(w) + g(w)
        """
        return self.obj_f(w)+self.g.obj(w)

    def grad_f(self,w):
        """
        compute gradient of loss nabla f(w)
        :param w: model parameter
        :return: nabla f(w)
        """
        return self.f.grad(w,self.data.x,self.data.y)


    def grad_prox(self, eta, w):
        """
        This function computes prox gradient of the objective f(w) + g(w)
        :param eta: learning rate
        :param w: parameter at which to compute proximal gradient
        :return: prox_grad(w) = (w- prox(w- eta* nabla f(w)))/eta
        """
        wt = w - eta * self.grad_f(w)
        ww = self.prox_map(eta, wt)
        return (w - ww) / eta

    def prox_map(self,eta,w,i=-1):
        """
        compute proximal map of g
        :param eta: learning rate
        :param w: model parameter
        :param i: the i-th component (-1 means all components)
        :return: argmin_u [ (0.5/eta)* || u-w||_2^2 + g_i(u) ]
        """
        return self.g.prox_map(eta,w,i)


class DecentralizedObj:
    """
     This class generates objective function for Problem 1
    """
    class _GenData:
        def __init__(self,n,d,seed):
            """
            The constructor, with fixed data generation
            """
            np.random.seed(seed)
            ntrn = n
            temp = np.ones((d, 1)) / np.linspace(1, 500, d).reshape((d, 1))
            wtrue = np.sqrt(temp)
            xtrn = np.random.rand(ntrn, d).dot(np.diagflat(temp))
            ptrn = 1 / (1 + np.exp(-xtrn.dot(wtrue)))
            ytrn = (np.random.rand(ntrn, 1) < ptrn) * 2 - 1
            self.x = xtrn
            self.y = ytrn

    def __init__(self,m,n,d):
        # a prime number
        q=939391
        seed=12345
        self.phi = []
        lam=1e-2
        mu=1e-2
        loss = LossLogistic()
        reg = RegL1L2(lam, mu)
        for i in range(m):
            train_data=DecentralizedObj._GenData(n,d,seed)
            seed = (seed * 2345 + 2721) % q
            phi=RegularizedLoss(train_data, loss, reg)
            self.phi.append(phi)
        self.m=m


    def grad(self,w):
        """
        This function computes the local gradient on all nodes
        :param w: d x m matrix, with column i the local vector at node i
        :return: d x m gradient matrix: column i the local gradient at node i
        """
        g=w.copy()
        for i in range(self.m):
            g[:,i]=self.phi[i].grad_f(w[:,i]).reshape(-1)
        return g

    def prox_map(self,eta,z):
        """
        This function computes the local prox_map on all nodes
        :param eta: learning rate
        :param z: d x m matrix, with column i the local vector at node i
        :return: d x m proximal mapping results: column i the local proximal mapping at node i
        """
        zp=z.copy()
        for i in range(self.m):
            zp[:,i]=self.phi[i].prox_map(eta,z[:,i])
        return zp

    def _send(self,i,w):
        """
        This function sends vector w on node i over network
        :param i: node number from 1 to m
        :param w: d x 1: vector to send
        :return:
        """
        self._network_vec[:,i]=w.copy().reshape(-1)

    def _receive(self,i):
        """
        This function receives on node i vectors from network
             assume that network contains vec(i) sent by node i for i=1 ... m
        :param i: node number from 1 to m
        :return: (vec(i-1) + vec(i+1))/2
        """
        m=self.m
        i1=(i+m-1)%m
        i2=(i+1)%m
        vec=(self._network_vec[:,i1]+self._network_vec[:,i2])/2
        return vec

    def communicate(self,vec):
        """
        This function simulates decentralized communication over network using self._send and self._receive
        :param vec: m x d parameter to communicate, the i-th column contains the vector of node i
        :return: m x d averaged vector i-th column is (vec[:,i-1]+vec[:,i+1])/2
        """
        self._network_vec=np.zeros((np.size(vec,0),np.size(vec,1)))
        for i in range(self.m):
            self._send(i,vec[:,i])
        vec2=vec.copy()
        for i in range(self.m):
            vec2[:,i]=self._receive(i)
        return vec2

class DecentralizedSolver:
    """
    Implement ADMM and accelerated linearized ADMM of Lecture 16
    """
    @staticmethod
    def solve_admm(phi,d,m,eta,rho,t):
        """
         solve min_x phi(x) = f(x) + g(z)   x = z using accelerated linearized ADMM
        :param phi: composite objective function to be optimized (require phi.grad_f() and phi.prox_map())
        :param eta: primal learning rate for f(x)
        :param beta: momentum parameter
        :param t: number of iterations
        :return: the t+1 iterates found by the method from z_0 to z_t
        """
        alpha = np.zeros((d,m))
        z= np.zeros((d,m))
        w=np.zeros((d,m))
        alphap=np.zeros((d,m))
        alphap2=np.zeros((d,m))
        result = np.zeros((d, t + 1))
        result[:, 0] = z[:,0].transpose()
        # implement
        
        return result

def prob1():
    # use mnist 1 versus 7 dataset
    train_data=BinaryClassificationData("mnist/mnist_train_binary.csv")
    test_data=BinaryClassificationData("mnist/mnist_test_binary.csv")

    # implement SDCA for 100 epochs
    # implement dual-free SDCA for 100 epochs
    # implement SGD for 100 epochs
    # plot convergence curves and save to prob1.pdf


def prob2():
    seed = 12345
    # solve the minimax problem in problem 2 with
    d=5
    np.random.seed(seed)
    b=np.random.random(d)
    A=np.random.random((d, d))

    # implement GDA for 100 iterations
    # implement extra gradient for 100 iterations
    # implement optimistic GDA for 100 iterations
    

def prob3():
    #   problem # 3
    seed = 12345
    # solve the objective function in problem 3 with
    d=10
    np.random.seed(seed)
    b=np.random.random(d)
    A=np.random.random((d, d))    
    c=1000
    # implement Example 18.10
    # implement Example 18.11
    

def prob4():
    # problem # 4
    m=5
    n=50
    d=200
    obj=DecentralizedObj(m,n,d)

    def check_convergence(obj,w):
        g=0
        for i in range(obj.m):
            g = g + obj.phi[i].grad_f(w)
        g=g/m
        g.shape=w.shape
        eta=0.1
        ww=obj.phi[0].prox_map(eta,w-eta*g)
        g=(w.reshape(-1)-ww.reshape(-1))/eta
        return np.linalg.norm(g,2)

    eta=0.5
    rho=0.5
    t=5000
    
    
    # implement an ADMM decentralized solver to solve obj by using obj.grad() obj.prox_map() and obj.communicate()
    

def main():
    prob1()
    prob2()
    prob3()
    prob4()

if __name__ == "__main__":
    main()
