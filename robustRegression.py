#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
from scipy.special import softmax

class RobustRegression(object):
    def __init__(self):
        pass

    def fit(self,X,y,iter=1000,lr=0.1):
        train_loss_list = []
        assert X.shape[0] == y.shape[0], 'mismatch number of samples and labels'
        self.N, self.n_feature = X.shape
        self.B = np.random.uniform(0,1,self.n_feature)
        self.X = X
        self.y = y
        for i in range(iter):
            # gradient step
            self.back_prop()
            self.B += -lr*self.G
            self.B = self.projsplx(self.B)
            if i % 10 == 1:
                loss = self.getLoss()
                print('iter: %s, loss: %s'%(i,loss))
                train_loss_list.append(loss)
        print(self.B,self.B.sum())
        return train_loss_list

    def fit_MD(self,X,y,iter=1000,lr=0.1):
        train_loss_list = []
        assert X.shape[0] == y.shape[0], 'mismatch number of samples and labels'
        self.N, self.n_feature = X.shape
        self.B = np.random.uniform(0,1,self.n_feature)
        self.X = X
        self.y = y
        for i in range(iter):
            self.back_prop()
            # Mirror Descent step
            # by solving min lr < gt,x > + D(x,xt) overl simplex
            # optimal point is normalized version of xt/exp(lr gt)
            self.B = self.B / np.exp(lr * self.G)
            self.B = np.abs(self.B) / np.linalg.norm(self.B,1)
            if i % 10 == 1:
                loss = self.getLoss()
                print('iter: %s, loss: %s'%(i,loss))
                #print(self.B)
                train_loss_list.append(loss)
        print(self.B,self.B.sum())
        return train_loss_list

    def projsplx(self,y):
        """projsplx projects a vector to a simplex
        by the algorithm presented in
        (Chen an Ye, "Projection Onto A Simplex", 2011)"""
        assert len(y.shape) == 1
        N = y.shape[0]
        y_flipsort = np.flipud(np.sort(y))
        cumsum = np.cumsum(y_flipsort)
        t = (cumsum - 1) / np.arange(1,N+1).astype('float')
        t_iter = t[:-1]
        t_last = t[-1]
        y_iter = y_flipsort[1:]
        if np.all((t_iter - y_iter) < 0):
            t_hat = t_last
        else:
        # find i such that t>=y
            eq_idx = np.searchsorted(t_iter - y_iter, 0, side='left')
            t_hat = t_iter[eq_idx]
        x = y - t_hat
        # there may be a numerical error such that the constraints are not exactly met.
        x[x<0.] = 0.
        x[x>1.] = 1.
        assert np.abs(x.sum() - 1.) <= 1e-5
        assert np.all(x >= 0) and np.all(x <= 1.)
        return x

    def getLoss(self):
        return np.linalg.norm(self.X.dot(self.B)-self.y)

    def back_prop(self):
        predict = self.X.dot(self.B) - self.y
        predict[predict>0] = 1
        predict[predict<0] = -1
        self.G = self.X.T.dot(predict)

if __name__ == '__main__':
    ITER = 2000
    X = np.load('data/X.npy')
    y = np.load('data/y.npy')
    model = RobustRegression()
    train_loss = model.fit(X,y,iter=ITER,lr=0.0001)
    train_loss_MD = model.fit_MD(X,y,iter=ITER,lr=0.0001)
    plt.plot(train_loss,'blue')
    plt.plot(train_loss_MD,'red')
    plt.title('loss')
    plt.savefig('5.pdf')
    plt.show()
