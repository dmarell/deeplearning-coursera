import numpy as np

parameter = {}
layers = (1,4,3,2,1)
print(len(layers))
for i in range(1, int(len(layers))):
    parameter['W' + str(i)] = np.random.randn(layers[i], layers[i-1]) * 0.01
    parameter['b' + str(i)] = np.random.randn(layers[i], 1) * 0.01
print('parameter:', parameter)
