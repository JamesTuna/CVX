#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
from scipy.special import softmax

# find a subgradient of ||A||*
def subgradient_nuclear_norm(A,epsilon=1e-5):
    m,n = A.shape
    U,S,VT = np.linalg.svd(A)
    V = VT.T
    # rank of A
    rank = (S > epsilon).sum()
    #print(rank)
    U1,U2 = U[:,:rank],U[:,rank:]
    V1,V2 = V[:,:rank],V[:,rank:]
    # generate a T with singular values uniformly distributed on [0,1]
    T_rank = min(m-rank,n-rank)
    T = np.random.uniform(0,1,T_rank)
    Im = np.identity(m-rank)
    In = np.identity(n-rank)
    T = Im.dot(np.diag(T)).dot(In)
    subgrad = U1.dot(V1.T) + U2.dot(T).dot(V2.T)

    return subgrad

def project(X,M,O):
    A = np.multiply(X,1-O) # for entries outside Ω
    B = np.multiply(M,O) # for entries in Ω
    return A+B

def optimize(X,M,O,step_size,iteration):
    loss = 0
    for i in range(iteration):
        subgrad = subgradient_nuclear_norm(X)
        X += - step_size * subgrad
        X = project(X,M,O)
        #loss = ((X-M)**2).sum()/10000
        #print(i,loss)
    loss = ((X-M)**2).sum()/10000
    return loss


if __name__ == '__main__':
    k_list = np.array([1,10,20,30,40,50,60,70,80,90,100])**2
    M = np.array(pd.read_csv('data/MatrixCompletion/M.csv',header=None))
    O = np.array(pd.read_csv('data/MatrixCompletion/O.csv',header=None))
    X = np.multiply(M,O)
    loss1_list = []
    loss2_list = []
    for k in k_list:
        loss1 = optimize(X,M,O,1/np.sqrt(k),k)
        loss2 = optimize(X,M,O,1/k,k)
        print(k,loss1,loss2)
        loss1_list.append(loss1)
        loss2_list.append(loss2)
    plt.plot(k_list,loss1_list)
    plt.plot(k_list,loss2_list)
    plt.savefig('MatrixCompletion.pdf')
    plt.show()
