import numpy as np

def Relu(X):
    return(np.maximum(X,0))

def deriv_relu(X):
    return(1 if X>0 else 0 )

def Softmax(X):
    e = np.exp(X - max(X))
    z = np.sum(e)
    y = e/z
    return y

def one_hot_encode(Y):
    oh = np.zeros((10,1))
    oh[int(Y), 0]=1
    return oh

def get_pred(A2):
    return np.argmax(A2,0)

def get_accu(pred, Y):
    return np.sum(pred == Y)/Y.size

def cross_entropy(y_true,y_pred):
    prob_correct = np.sum(y_true*y_pred)
    loss = -np.log(prob_correct)
    return loss