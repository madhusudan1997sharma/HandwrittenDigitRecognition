import numpy as np
import pandas as pd

def sigmoid(m):
    return 1/(1 + np.exp(-m))

def derivative_sigmoid(m):
    return m * (1 - m)

raw = pd.read_csv("Dataset.csv").values

epoch = 2000
lr = 0.01
w_hidden = np.random.uniform(low = -4, high = 4, size = (784, 128))
w_out = np.random.uniform(low = -1, high = 1, size = (128, 10))
b_hidden = np.random.uniform(low = -1, high = 1, size = (1, 128))
b_out = np.random.uniform(low = -1, high = 1, size = (1, 10))

batches = 200
inc = 200
fr = 0
t = inc
for j in range(batches):
    data = raw[fr:t,1:]
    out = raw[fr:t,0]
    size = raw[fr:t,0].size
    result = np.zeros((size, 10), dtype=int)
    for i in range(size):
        result[i][out[i]] = 1
    x = data/255    # CONVERTING 0-255 TO 0-1
    y = result
    
    for i in range(epoch):
        hiddenlayer_in = np.dot(x, w_hidden) + b_hidden
        hiddenlayer_out = sigmoid(hiddenlayer_in)
        outputlayer_in = np.dot(hiddenlayer_out, w_out) + b_out
        output = sigmoid(outputlayer_in)
        gradient_outputlayer = derivative_sigmoid(output)
        gradient_hiddenlayer = derivative_sigmoid(hiddenlayer_out)
        error_outputlayer = y - output
        d_outputlayer = error_outputlayer * gradient_outputlayer
        error_hiddenlayer = d_outputlayer.dot(w_out.T)
        d_hiddenlayer = error_hiddenlayer * gradient_hiddenlayer
        w_out += hiddenlayer_out.T.dot(d_outputlayer) * lr
        b_out = b_out + np.sum(d_outputlayer, axis = 0, keepdims = True) * lr
        w_hidden += x.T.dot(d_hiddenlayer) * lr
        b_hidden = b_hidden + np.sum(d_hiddenlayer, axis = 0, keepdims = True) * lr

        print("Epoch: ",i,"/",epoch)

    fr += inc
    t += inc

print("Training Complete. Weights and Bias CSVs are saved.")
np.savetxt("WHidden.csv", w_hidden, delimiter=",")
np.savetxt("BHidden.csv", b_hidden, delimiter=",")
np.savetxt("WOutput.csv", w_out, delimiter=",")
np.savetxt("BOutput.csv", b_out, delimiter=",")
