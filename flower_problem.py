import matplotlib
import numpy as np
# Goal: Find what colour corresponds to the length and width of a flower petal
# Input: [4.5, 1, ?]
# My Data L,W,C (0 or 1)
data = [[3, 1.5, 1],
        [2, 1, 0],
        [4, 1.5, 1],
        [3, 1, 0],
        [3.5, .5, 1],
        [2, .5, 0],
        [5.5, 1, 1],
        [1, 1, 0]]

unknownFlower = [4.5, 1]

# Network Architecture

#    O   <--  Missing Flower Colour
#  /  \  <-- weight1, weight2, bias
# O   O  <-- Inputs (Width and Length)
w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

# Sigmoid allows for number between 0 and 1 to decide flower colour

def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Derivative of Sigmoid (Bell Curve)
def sigmoidPrime(x):
    return sigmoid(x) * (1-sigmoid(x))

# Training
for i in range(5000):
    ri = np.random.randint(len(data))
    point = data[ri]
    print(point)
