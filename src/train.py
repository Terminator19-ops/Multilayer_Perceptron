from src.model import init_params, forward_prop, back_prop, update
from src.utils import one_hot_encode, cross_entropy

def gradient_descent(X, Y, batch_size, alpha, iter):
    W1, B1, W2, B2 = init_params()
    m, n = X.shape
    if n % batch_size == 0:
        flag = True
    else:
        flag = False
    
    sdw1, sdw2, sdb1, sdb2 = 0,0,0,0

    for _ in range(iter):
        if(flag):
            for j in range(n):
                c_x = (X[:,j]).reshape(784,1)
                label = int(Y[:,j])
                oh = one_hot_encode(label)
                A1,A2,Z1 = forward_prop(c_x,W1,B1,W2,B2)
                DW1,DW2,DB1,DB2 = back_prop(c_x,oh,A1,A2,Z1,W2)
                sdw1 += DW1
                sdw2 += DW2
                sdb1 += DB1
                sdb2 += DB2
                if (j+1)%batch_size == 0:
                    sdw1 = sdw1/batch_size
                    sdw2 = sdw2/batch_size
                    sdb1 = sdb1/batch_size
                    sdb2 = sdb2/batch_size
                    W1,W2,B1,B2 = update(W1,W2,B1,B2,sdw1,sdw2,sdb1,sdb2,alpha)
                    sdw1, sdw2, sdb1, sdb2 = 0,0,0,0

                x2 = cross_entropy(oh, A2)
                if j == batch_size:
                    print(x2) 

    return W1,W2,B1,B2