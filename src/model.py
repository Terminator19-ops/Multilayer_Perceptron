from src.utils import Relu, Softmax
import numpy as np


def init_params():
    w1 = np.random.rand(10,784) - 0.5
    b1 = np.random.rand(10,1) - 0.5
    w2 = np.random.rand(10,10) - 0.5
    b2 = np.random.rand(10,1) - 0.5
    return w1,b1,w2,b2

def forward_prop(x, w1, b1, w2, b2):
    z1 = w1.dot(x) + b1
    a1 = Relu(z1)   
    z2 = w2.dot(a1) + b2
    a2 = Softmax(z2)

    return a1, a2, z1, z2

def back_prop(x, y, a1, a2, z1, w2):
    dz2 = a2 - y
    dw2 = (dz2).dot(a1.T)
    db2 = dz2
    dz1 = (w2.T).dot(dz2) * (z1 > 0)
    dw1 = dz1.dot(x.T)
    db1 = dz1
    return dw1, dw2, db1, db2

def update(w1, w2, b1, b2, dw1, dw2, db1, db2, alpha):
    w1 = w1 - alpha*dw1
    w2 = w2 - alpha*dw2
    b1 = b1 - alpha*db1
    b2 = b2 - alpha*db2

    return w1, w2, b1, b2