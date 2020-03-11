import matplotlib
from matplotlib import pyplot as plt
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

# Scatter plot
plt.axis([0, 7, 0, 7])
plt.grid()
for i in range(len(data)):
    point = data[i]
    colour = "r"
    if point[2] == 0:
        colour = "b"
    plt.scatter(point[0], point[1], c = colour)

plt.show()

# Training
learning_rate = .2
costs = []

for i in range(500000):
    ri = np.random.randint(len(data))
    point = data[ri]

    z = point[0] * w1 + point[1] * w2 + b
    pred = sigmoid(z)

    target = point[2]
    cost = np.square(pred - target)

    # Cost Updates
    costs.append(cost)

    # Derivative of cost with respect to prediction
    dcost_pred = 2 * (pred - target)

    # Derivative of prediction with respect to z
    dpred_dz = sigmoidPrime(z)

    # Derivative of z with respect to Length and Width
    dz_dw1 = point[0]
    dz_dw2 = point[1]
    dz_db = 1

    # Derivative of cost with respect to parameters
    dcost_dz = dcost_pred * dpred_dz
    dcost_dw1 = dcost_dz * dz_dw1
    dcost_dw2 =  dcost_dz * dz_dw2
    dcost_db =  dcost_dz * dz_db

    # Fraction of parameters

    w1 = w1 - learning_rate * dcost_dw1
    w2 = w2 - learning_rate * dcost_dw2
    b = b - learning_rate * dcost_db

    if i % 100 == 0:
        cost_sum = 0
        for j in range(len(data)):
            point = data[ri]
            z = point[0] * w1 + point[1] * w2 + b
            pred = sigmoid(z)
            target = point[2]
            cost_sum += np.square(pred - target)
        costs.append(cost_sum/len(data))

# Prediction of unknownFlower
z = unknownFlower[0] * w1 + unknownFlower[1] * w2 + b
pred = sigmoid(z)
print(pred)

# Plot
plt.plot(costs)
plt.show()
