import scipy.io
import numpy as np

def load_mnist(trainpath,testpath):

    traindata = scipy.io.loadmat(trainpath)
    testdata = scipy.io.loadmat(testpath)

    train_data = traindata['trainData'][0][0]
    for i in range(9):
        train_data = np.concatenate((train_data,traindata['trainData'][i+1][0]),axis = 0)

    test_data = testdata['testData'][0][0]
    for j in range(9):
        test_data = np.concatenate((test_data,testdata['testData'][j+1][0]),axis = 0)

    train_labels = []
    test_labels = []

    for i in range(10):
        for j in range(traindata['trainData'][i][0].shape[0]):
            train_labels.append(i)

    for k in range(10):
        for m in range(testdata['testData'][k][0].shape[0]):
            test_labels.append(k)

    train_labels = np.asarray(train_labels) 
    test_labels = np.asarray(test_labels) 
    
    return train_data, train_labels, test_data, test_labels

def load_SampleMnist(trainpath,testpath,nSamples):
    
    traindata = scipy.io.loadmat(trainpath)
    testdata = scipy.io.loadmat(testpath)

    train_data = traindata['trainData'][0][0][0:nSamples]
    
    for i in range(9):
        train_data = np.concatenate((train_data,traindata['trainData'][i+1][0][0:nSamples]),axis = 0)

    test_data = testdata['testData'][0][0]
    
    for j in range(9):
        test_data = np.concatenate((test_data,testdata['testData'][j+1][0]),axis = 0)

    train_labels = []
    test_labels = []

    for i in range(10):
        for j in range(traindata['trainData'][i][0][0:nSamples].shape[0]):
            train_labels.append(i)

    for k in range(10):
        for m in range(testdata['testData'][k][0].shape[0]):
            test_labels.append(k)

    train_labels = np.asarray(train_labels) 
    test_labels = np.asarray(test_labels) 
    
    return train_data, train_labels, test_data, test_labels
