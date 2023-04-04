#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np 
from scipy.stats import multivariate_normal as mvn
from mnist import load_SampleMnist
from mnist import load_mnist

class Gauss_class_diag():
    """
    This class implements a Gaussian classifier with diagonal covariance matrices 
    """
    def fit(self, X, y, epsilon = 0.5e-1): 
        self.stats = dict()
        self.priors = dict()
        self.labels = set(y.astype(int))    # Unique class labels: 0,1,2,...,K-1 

        for k in self.labels:    
            X_k = X[y==k,:]     # Select data from the k-th class
            self.stats[k] = {"mean":X_k.mean(axis=0), "cov":X_k.var(axis=0) + epsilon }
            self.priors[k]=len(X_k)/len(X)

    def predict(self, X):
        N, D = X.shape
        P_hat = np.zeros((N,len(self.labels)))
        for k, s in self.stats.items():
            P_hat[:,k] = mvn.logpdf(X, s["mean"], s["cov"]) + np.log(self.priors[k]) 
  
        return P_hat.argmax(axis=1)


class Gauss_class_full():
    """
    This class implements a Gaussian classifier with full covariance matrices
    """
    def fit(self, X,y, epsilon=0.5e-1):
        self.stats = dict()
        self.priors = dict()
        self.labels = set(y.astype(int))
        
        for k in self.labels:
            X_k = X[y==k,:]
            N_k,D = X_k.shape   # N_k=total number of observations of that class
            mu_k = X_k.mean(axis=0)
            self.stats[k] = {"mean":X_k.mean(axis=0), 
                              "cov": (1/(N_k-1))*np.matmul((X_k-mu_k).T, X_k-mu_k) + 
                             epsilon*np.identity(D)}
            self.priors[k] = len(X_k)/len(X)
    
    
    def predict(self, X):
        N,D = X.shape
        P_hat = np.zeros((N,len(self.labels)))
        
        for k,s in self.stats.items():
            P_hat[:,k] = mvn.logpdf(X, s["mean"], s["cov"]) + np.log(self.priors[k])
        
        return P_hat.argmax(axis=1)

# Compute accuracy
def accuracy(y, y_hat):
    return np.mean(y==y_hat)

option = 1

if option == 1:
    # Load noisy samples
    trainpath = '../data/noisy_train_digits.mat'
    testpath = '../data/noisy_test_digits.mat'
    train_data, train_labels, test_data, test_labels= load_mnist(trainpath, testpath)
elif option == 2:
    # Load N noisy training samples and all test samples
    trainpath = '../data/noisy_train_digits.mat'
    testpath = '../data/noisy_test_digits.mat'
    nSamples = 100
    train_data, train_labels, test_data, test_labels = load_SampleMnist(trainpath,testpath,nSamples)
elif option == 3:
    # Load clean samples
    trainpath = '../data/clean_train_digits.mat'
    testpath = '../data/clean_test_digits.mat'
    train_data, train_labels, test_data, test_labels= load_mnist(trainpath, testpath)
else:
    sys.exit("Wrong option")

# Create a Gaussian classifier with diag convariance matrices
gc = Gauss_class_diag()

# Create a Gaussian classifier with full convariance matrices
#gc = Gauss_class_full()

# Train the Gaussian classifier
gc.fit(train_data, train_labels, epsilon=0)

# Predict the class labels of test samples
y_test = gc.predict(test_data)

# Test accuracy
print(accuracy(test_labels, y_test))