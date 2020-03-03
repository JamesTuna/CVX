#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
from scipy.special import softmax

class LogisticRegression(object):

    def __init__(self,n_feature,n_class,regularization = 0):
        self.n_feature = n_feature
        self.n_class = n_class
        self.regularization = regularization

    def fit(self,X,y,optimizer='GD',iter=1e+4,lr=1e-3,batch=32,log_path='./',log_interval=2000):
        print("start training model...")
        assert (X.shape[1]==self.n_feature), 'mismatch: number of features'
        assert (y.max()<=self.n_class-1), 'mismatch: max index of classes'
        assert (y.min()>=0), 'mismatch: min index of classes'

        self.N = X.shape[0]
        self.X = X
        self.y = y
        self.y_onehot = np.eye(self.n_class)[self.y]
        self.B = np.random.normal(0,0.1,(self.n_feature,self.n_class))

        test_loss_list = []
        train_loss_list = []
        test_acc_list = []
        train_acc_list = []

        # parameters for nesterov
        AGD_lambda_old = 0
        AGD_lambda_new = None
        AGD_gamma = None
        AGD_y_old = self.B
        AGD_y_new = None

        # log data
        min_loss_test = np.inf
        min_loss_iter = 0
        min_loss_acc = 0
        file_old = None

        for i in range(iter):
            if optimizer == 'GD':
                self.back_prop(batch=batch)
                self.B += lr * (-self.G)
            elif optimizer == 'AGD':
                self.back_prop(batch=batch)
                AGD_y_new = self.B - lr * self.G
                AGD_lambda_new = (1+np.sqrt(1+4*AGD_lambda_old**2))/2
                AGD_gamma = (1-AGD_lambda_old)/AGD_lambda_new
                self.B = (1-AGD_gamma) * AGD_y_new + AGD_gamma * AGD_y_old

                AGD_y_old = AGD_y_new
                AGD_lambda_old = AGD_lambda_new

            # test and record
            if i % log_interval == 0:
                loss,acc = self.getLoss(self.X,self.y,self.B,0)
                test_loss,test_acc = self.getLoss(self.X_test,self.y_test,self.B,0)
                print('iter: %s train: loss: %.4f test_loss: %.4f'%(i,loss,test_loss))
                #print('iter: %s on test: loss: %.4f acc: %.4f'%(i,test_loss,test_acc))

                test_loss_list.append(test_loss)
                train_loss_list.append(loss)
                test_acc_list.append(test_acc)
                train_acc_list.append(acc)

                if test_loss < min_loss_test:
                    min_loss_test = test_loss
                    min_loss_iter = i
                    min_loss_acc = test_acc

                    '''
                    filename = '-'.join([optimizer,str(self.regularization),str(batch),str(lr),'loss',str(min_loss_test),'acc',str(min_loss_acc)])+'.npy'
                    if file_old != None:
                        os.remove(file_old)
                    np.save(filename,self.B)
                    file_old = filename
                    '''

        return train_loss_list,train_acc_list,test_loss_list,test_acc_list



    def getLoss(self,X,y,B,regularization = 0):
        N = X.shape[0]
        scores = - X.dot(B) # N x n_class
        prob_matrix = np.zeros((N,self.n_class))
        for i in range(N):
            for j in range(self.n_class):
                prob_i_j = np.exp(scores[i,j])/np.exp(scores[i,:]).sum()
                prob_matrix[i,j] = prob_i_j
        loss = 0
        for i in range(N):
            loss += -scores[i,y[i]]
            loss += np.log(np.exp(scores[i,:]).sum())
        loss /= N
        loss += regularization * ((B ** 2).sum())
        predict = prob_matrix.argmax(axis=1)
        accuracy = (predict == y[:N]).sum()/N
        return loss,accuracy

    def back_prop(self,batch=32):
        batch_samples = np.random.choice(self.N,batch,replace=False)
        # compute gradient
        Xs = self.X[batch_samples]
        onehot_ys = self.y_onehot[batch_samples]
        class_scores = - Xs.dot(self.B) # N x n_class
        probs = softmax(class_scores,axis=1)
        self.G = Xs.T.dot(onehot_ys - probs)
        # approximate expectation
        self.G /= batch
        # add graient of regularization term
        self.G += 2 * self.regularization * self.G

    def predict(self,X_test):
        print("not implemented")




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='logistic regression')
    parser.add_argument('--batch-size', type=int, default=12564, help='Number of samples per mini-batch')
    parser.add_argument('--iter', type=int, default=1e+4, help='Number of iteration to optimize')
    parser.add_argument('--lr', type=float, default=10, help='step size')
    parser.add_argument('--MU', type=float, default=0, help='regularization coefficient')
    parser.add_argument('--optimizer', type=str, default=0, help='GD or AGD')
    parser.add_argument('--interval', type=int, default=0, help='log interval')
    args = parser.parse_args()

    BATCH_SIZE = int(args.batch_size)
    LR = float(args.lr)
    MU = float(args.MU)
    ITER = int(args.iter)
    INTERVAL = int(args.interval)

    # read data
    X = np.array(pd.read_csv('data/logistic_news/X_train.csv',header=None))
    y = np.array(pd.read_csv('data/logistic_news/y_train.csv',header=None))[0]
    X_test = np.array(pd.read_csv('data/logistic_news/X_test.csv',header=None))
    y_test = np.array(pd.read_csv('data/logistic_news/y_test.csv',header=None))[0]
    # run SGD
    model = LogisticRegression(n_feature=X.shape[1],n_class=20,regularization = MU)
    model.X_test = X_test
    model.y_test = y_test
    train_loss,train_acc,test_loss,test_acc = model.fit(X,y,optimizer=args.optimizer,iter=ITER,lr=LR,batch=BATCH_SIZE,log_interval=INTERVAL)

    # draw plots
    print('train loss %.4f test loss %.4f'%(train_loss[-1],test_loss[-1]))
    iterations = np.array([i for i in range(0,ITER*2,INTERVAL)])
    iterations = iterations[:len(train_acc)]
    fig,axs = plt.subplots(1,2,figsize=(16,8))
    l1, = axs[0].plot(iterations,train_loss,'blue')
    l3, = axs[0].plot(iterations,test_loss,'red')
    iterations = iterations[:len(train_acc)]
    axs[0].set_title('loss')
    axs[0].set_xlabel('iteration')
    axs[0].set_ylabel('loss')


    l2, = axs[1].plot(iterations,train_acc,'blue')
    l4, = axs[1].plot(iterations,test_acc,'red')
    axs[1].set_title('accuracy')
    axs[1].set_xlabel('iteration')
    axs[1].set_ylabel('accuracy')

    axs[0].legend([l1,l3],['train','test'],loc='best')
    axs[1].legend([l2,l4],['train','test'],loc='best')
    plt.savefig('-'.join([args.optimizer,str(MU),str(BATCH_SIZE),str(ITER)])+'.pdf')
    plt.show()
