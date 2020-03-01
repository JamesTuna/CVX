#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
A = np.load('./data/A_train.npy')
b = np.load('./data/b_train.npy')

A_test = np.load('./data/A_test.npy')
b_test = np.load('./data/b_test.npy')

ATA = A.T.dot(A)
ATb = A.T.dot(b)
lipschitz = np.linalg.svd(ATA)[1][0] # smoothness of g()

def subGrad(iter,lr,regularization):
    x = np.random.normal(0,0.1,A.shape[1])
    iter = int(iter)
    loss1 = []
    loss2 = []
    for i in range(iter):
        # compute a subgradient
        g = np.zeros(x.shape[0])
        g[x>0] = 1
        g[x<0] = -1
        g *= regularization
        g += ATA.dot(x) - ATb
        x -=  lr * g
        # calculate loss
        loss_train = np.linalg.norm(A.dot(x)-b,2)
        loss_test = np.linalg.norm(A_test.dot(x)-b_test,2)
        print(i,' :train loss %.4f, test loss %.4f'%(loss_train,loss_test))
        loss1.append(loss_train)
        loss2.append(loss_test)
    return range(iter),loss1,loss2

def Pl(x,lr,regularization):
    # Pl() operator in paper Breck's FISTA paper
    # compute a subgradient of g(x)
    g = ATA.dot(x) - ATb
    x -= lr *g
    # soft threshold
    eta = lr*regularization
    x[np.abs(x)<=eta] = 0
    x[x>eta] = x[x>eta] - eta
    x[x<-eta] = x[x<-eta] + eta
    return x

def ISTA(iter,lr,regularization):
    x = np.random.normal(0,0.1,A.shape[1])
    iter = int(iter)
    loss1 = []
    loss2 = []
    lipschitz = np.linalg.svd(ATA)[1][0]
    for i in range(iter):
        x = Pl(x,lr,regularization)
        # calculate loss
        loss_train = np.linalg.norm(A.dot(x)-b,2)
        loss_test = np.linalg.norm(A_test.dot(x)-b_test,2)
        print(i,' :train loss %.4f, test loss %.4f'%(loss_train,loss_test))
        loss1.append(loss_train)
        loss2.append(loss_test)
    return range(iter),loss1,loss2

def FISTA(iter,lr,regularization):
    old_x = np.random.normal(0,0.1,A.shape[1])
    old_y = old_x.copy()
    old_t = 1
    iter = int(iter)
    loss1 = []
    loss2 = []
    lipschitz = np.linalg.svd(ATA)[1][0]

    for i in range(iter):
        new_x = Pl(old_y,lr,regularization)
        new_t = (1+np.sqrt(1+4*old_t**2))/2
        new_y = new_x + (new_x - old_x)*(old_t-1)/(new_t)
        # calculate loss
        x = new_y
        loss_train = np.linalg.norm(A.dot(x)-b,2)
        loss_test = np.linalg.norm(A_test.dot(x)-b_test,2)
        print(i,' :train loss %.4f, test loss %.4f'%(loss_train,loss_test))
        loss1.append(loss_train)
        loss2.append(loss_test)
        # update x,y,t
        old_x,old_y,old_t = new_x,new_y,new_t
    return range(iter),loss1,loss2

def FrankWolfe(iter,lr,regularization):
    k = 1000
    x = np.random.normal(0,0.1,A.shape[1])
    iter = int(iter)
    loss1 = []
    loss2 = []
    for i in range(iter):
        # compute a subgradient
        g = np.zeros(x.shape[0])
        g[x>0] = 1
        g[x<0] = -1
        g *= regularization
        g += ATA.dot(x) - ATb
        # minimize gty over unit k ball
        print(np.linalg.norm(g))
        y = - k * g / np.linalg.norm(g)
        x = x * (1-lr) + y * lr

        loss_train = np.linalg.norm(A.dot(x)-b,2)
        loss_test = np.linalg.norm(A_test.dot(x)-b_test,2)
        print(i,' :train loss %.4f, test loss %.4f'%(loss_train,loss_test))
        loss1.append(loss_train)
        loss2.append(loss_test)
    return range(iter),loss1,loss2







if __name__ == '__main__':
    iter = 1e+4
    #iter = 0
    #subGrad(1,1e-4,20)
    fig, axs = plt.subplots(1,2,sharex=True)
    x_guess = np.linalg.inv(ATA).dot(A.T).dot(b)
    print('x_guess\n',x_guess)
    print('test loss',np.linalg.norm(A_test.dot(x_guess)-b_test,2))
    print('train loss',np.linalg.norm(A.dot(x_guess)-b,2))

    i,loss_train,loss_test = subGrad(iter,1e-6,20)
    axs[0].plot(i,loss_train,'blue')
    axs[0].set_title('Training loss')
    axs[1].plot(i,loss_test,'blue')
    axs[1].set_title('Test loss')
    i,loss_train,loss_test = ISTA(iter,1e-6,20)
    axs[0].plot(i,loss_train,'red')
    axs[0].set_title('Training loss')
    axs[1].plot(i,loss_test,'red')
    axs[1].set_title('Test loss')
    i,loss_train,loss_test = FISTA(iter,1e-6,20)
    axs[0].plot(i,loss_train,'black')
    axs[0].set_title('Training loss')
    axs[1].plot(i,loss_test,'black')
    axs[1].set_title('Test loss')

    i,loss_train,loss_test = FrankWolfe(iter,1e-6,20)
    axs[0].plot(i,loss_train,'green')
    axs[0].set_title('Training loss')
    axs[1].plot(i,loss_test,'green')
    axs[1].set_title('Test loss')
    plt.show()
