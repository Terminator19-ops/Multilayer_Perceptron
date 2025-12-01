import numpy as np

def Relu(X):
    return(np.maximum(X,0))

def deriv_relu(X):
    return(1 if X>0 else 0 )

def Softmax(X):
    e = np.exp(X -255)
    return e/sum(e)

def one_hot_encode(Y):
    oh = np.zeros((10,1))
    oh[int(Y), 0]=1
    return oh

def get_pred(A2):
    return np.argmax(A2,0)

def get_accu(pred, Y):
    return np.sum(pred == Y)/Y.size

def cross_entropy(y_true,y_pred):
    y_max = np.zeros((10500,1))
    i = 0
    for x in y_pred:
        t = max(x)
        y_max[i] = t
        i += 1
    ln_y = np.log(y_max)
    loss = np.sum(y_true*ln_y)
    return loss