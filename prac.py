import numpy as np

tx_bits = np.random.randint(2, size=(14, 1))  # data bits for one block






b= [2,3]
a = np.ones(shape=(1,2))

print(np.ones(shape=(10,)))
print(np.kron(a,b).reshape((2,-1)).shape)