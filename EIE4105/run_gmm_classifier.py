#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from mnist import load_mnist
from classifier import classifier
import numpy as np

# Compute accuracy
def accuracy(y, y_hat):
    return np.mean(y==y_hat)

# Load data
trainpath = '../data/noisy_train_digits.mat'
testpath = '../data/noisy_test_digits.mat'
#train_data, train_labels, test_data, test_labels= load_mnist(trainpath, testpath)

# Load 100 training samples
from mnist import load_SampleMnist
nSamples = 6000
train_data, train_labels, test_data, test_labels = load_SampleMnist(trainpath,testpath,nSamples)

# Train a full-cov GMM classifier
M = 2
gmm_cls = classifier(M, model_type='gmm', covariance_type='full', verbose=False)
gmm_cls.fit(train_data, train_labels)

# Predict the labels of the training data
y_train = gmm_cls.predict(train_data)

# Accuracy of GMM classifier on train data
print(f"Train accuracy = {accuracy(train_labels, y_train)}")

# Predict the labels of the test data
y_test = gmm_cls.predict(test_data)

# Accuracy of GMM classifier on test data
print(f"Test accuracy = {accuracy(test_labels, y_test)}")