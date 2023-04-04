import numpy as np

tx_bits = np.random.randint(2, size=(14, 1))  # data bits for one block

x = []

for i in range(1,4):
    x.append([i, i+1])

print(x)