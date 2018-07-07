import numpy as np
import pandas as pd

def result_cal(m):
    return np.where(m > 0.85)[0]

def sigmoid(m):
    return 1/(1 + np.exp(-m))

def relu(x):
    y=x
    for i in range(x.shape[0]):
        y[i]=np.maximum(0,x[i])
    return y

raw = pd.read_csv("Dataset.csv").values
    
w_hidden = np.genfromtxt("WHidden.csv", delimiter=",")
b_hidden = np.genfromtxt("BHidden.csv", delimiter=",")
w_out = np.genfromtxt("WOutput.csv", delimiter=",")
b_out = np.genfromtxt("BOutput.csv", delimiter=",")

batch = 100
fr = 41500
t = fr + batch
data = raw[fr:t,1:]
out = raw[fr:t,0]
size = raw[fr:t,0].size
result = np.zeros((size, 10), dtype=int)
for i in range(size):
    result[i][out[i]] = 1

print("Input Test Data:")
for i in range(size):
    print(result_cal(result[i]))

x = data/255
hiddenlayer_in = np.dot(x, w_hidden) + b_hidden
hiddenlayer_out = sigmoid(hiddenlayer_in)
outputlayer_in = np.dot(hiddenlayer_out, w_out) + b_out
output = relu(outputlayer_in)
f  = 0
t = 0
print("Output:")
for i in range(size):
    status = result_cal(result[i]) == result_cal(output[i])
    if status.size == 1 and status[0] == True:
        t+=1
    else:
        f+=1
    print("Match Status: ", status)

print("Accuracy: ", (t/batch)*100, "%")
