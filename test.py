import numpy as np
import pandas as pd
from src.model import forward_prop
from src.utils import get_pred

data = np.load('checkpoints/model_weights.npz')
w1 = data["W1"]
w2 = data["W2"]
b1 = data["B1"]
b2 = data["B2"]
 
test_df = pd.read_csv('data/test.csv')
m, n = test_df.shape
test_labels = test_df.iloc[:,0:1]
test_pixels = test_df.iloc[:,1:n]

test_labels = np.array(test_labels)
test_pixels = np.array(test_pixels)

accu = 0
for i in range(m):
    shaped_test_pixel = test_pixels[i].reshape(784,1)
    a1,a2,z1 = forward_prop(shaped_test_pixel,w1,b1,w2,b2)
    x1 = get_pred(a2)
    if (test_labels[i]==x1):
        accu = accu + 1

accu = accu/m
print(accu)


    
    
    
