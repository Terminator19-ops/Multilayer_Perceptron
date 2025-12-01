from .model import init_params, forward_prop, back_prop, update
from .utils import one_hot_encode

def gradient_descent(X, Y, Y_test, alpha, iter):
    W1, B1, W2, B2 = init_params()
    m, n = X.shape
    k2 = X.max()
    for ik in range(iter):
        for j in range(n):
            c_x = (X[:,j]).reshape(784,1)
            label = int(Y[:,j])
            oh = one_hot_encode(label)
            A1,A2,Z1,Z2 = forward_prop(c_x,W1,B1,W2,B2)
            DW1,DW2,DB1,DB2 = back_prop(c_x,oh,A1,A2,Z1,W2)
            W1,W2,B1,B2 = update(W1,W2,B1,B2,DW1,DW2,DB1,DB2,alpha)   
            
        if ik %20 == 0:
                print(cross_entropy(Y_test, A2)) # type: ignore 
#print(crossentropy(gradient_descent(in range (back))))

    return W1,W2,B1,B2