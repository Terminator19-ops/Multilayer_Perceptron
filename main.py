import pandas as pd
import numpy as np
from .src.train import gradient_descent
from .src.model import forward_prop
from .src.utils import cross_entropy

df = pd.read_csv('data/train.csv')
df = np.array(df)
m, n = df.shape
np.random.shuffle(df)

train_data = df[0:int(0.75*m)]
val_data = df[int(0.75*m):m]



X_train = train_data[:,1:n+1]
X_train = X_train/255
X_train = X_train.T
y_train = train_data[:,0:1]
y_train = y_train.T
X_test = val_data[:,1:n+1]
X_test = X_test/255
X_test = X_test.T
y_test = val_data[:,1]
y_test = y_test.T


W1, W2, B1, B2 = gradient_descent(X_train, y_train, y_test, 0.01,100)

a1, a2, z1, z2 = forward_prop(X_test,W1,B1,W2,B2)
a2 = a2.T
for x in a2:
    print(x)
a2.shape


y_test = y_test.reshape((10500,1))
loss = cross_entropy(y_test,a2)
print(loss)
