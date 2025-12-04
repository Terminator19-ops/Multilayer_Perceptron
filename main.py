import pandas as pd
import numpy as np
from src.train import gradient_descent
from src.model import forward_prop
from src.utils import cross_entropy

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
y_test = y_test.reshape(10500,1)



W1, W2, B1, B2 = gradient_descent(X_train, y_train,100, 0.0001,500)

np.savez("checkpoints/model_weights.npz",W1 = W1, W2 = W2, B1 = B1, B2 = B2)
