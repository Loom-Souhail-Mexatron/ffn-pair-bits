import numpy as np
from random import random as rd

def activation_func(x):
    return 1 / (1 + np.exp(-x))#np.tanh(x)

def activation_func_deriv(x):
    return activation_func(x) * (1 - activation_func(x))

def weightsGen(x):
    return rd() * 2 - 1

def weightArr(i, j):
    arr = np.array(i * [j * [float()]])
    for y in range(i):
        for x in range(j):
            arr[y][x] = weightsGen(arr[y][x])
    return arr

data = {
    'X': [[0, 0], [0, 1], [1, 0], [1, 1]],
    'Y': [[1, 1], [1, 0], [0, 1], [0, 0]]
}

inpData = data['X']
outData = data['Y']

dataSize = len(inpData)
input_nodes = len(inpData[0])
output_nodes = len(outData[0])
hidden_nodes = 5
learning_rate = 0.1
epochs = 1000

weights_ih = weightArr(input_nodes, hidden_nodes)
weights_ho = weightArr(hidden_nodes, output_nodes)

print("# INPUT NODES:",input_nodes)
print("# OUTPUT NODES:",output_nodes)
print("# HIDDEN NODES:",hidden_nodes)
print("LEARNING RATE:",learning_rate)

# Train phase
for epoch in range(epochs):
    if epoch%100==0:
        perc = int(epoch*100/epochs)
        print("EPOCH",epoch,"OUT OF",epochs,';',str(perc)+'% DONE')
        print('|',end='')
        print("="*perc,end='')
        print('-'*(100-perc),end='')
        print('|',end='')
        print('')
    for i in range(dataSize):
        # Forward Pass
        hidden = np.array(hidden_nodes * [float()])
        for j in range(hidden_nodes):
            sum_ = 0
            for k in range(input_nodes):
                sum_ += inpData[i][k] * weights_ih[k][j]
            hidden[j] = activation_func(sum_)

        output = np.array(output_nodes * [float()])
        for o in range(output_nodes):
            for j in range(hidden_nodes):
                output[o] += hidden[j] * weights_ho[j][o]
            output[o] = activation_func(output[o])

        # Backward Pass
        delOutput = np.empty_like(output)
        for o in range(output_nodes):
            delOutput[o] = (outData[i][o] - output[o]) * activation_func_deriv(output[o])

        delHidden = np.empty_like(hidden)
        for j in range(hidden_nodes):
            error = 0
            for o in range(output_nodes):
                error += delOutput[o] * weights_ho[j][o]
            delHidden[j] = error * activation_func_deriv(hidden[j])

        for k in range(input_nodes):
            for j in range(hidden_nodes):
                weights_ih[k][j] += learning_rate * inpData[i][k] * delHidden[j]

        for j in range(hidden_nodes):
            for o in range(output_nodes):
                weights_ho[j][o] += learning_rate * hidden[j] * delOutput[o]

print('')
# Test phase
for i in range(dataSize):
    hidden = np.array(hidden_nodes * [float()])
    for j in range(hidden_nodes):
        sum_ = 0
        for k in range(input_nodes):
            sum_ += inpData[i][k] * weights_ih[k][j]
        hidden[j] = activation_func(sum_)

    output = np.array(output_nodes * [float()])
    for o in range(output_nodes):
        for j in range(hidden_nodes):
            output[o] += hidden[j] * weights_ho[j][o]
        output[o] = activation_func(output[o])
        output[o] = 1 if output[o] > 0.5 else 0

    print("INPUT:", inpData[i][0], ",", inpData[i][1],end=' ')
    print("OUTPUT:", output[0], output[1])

