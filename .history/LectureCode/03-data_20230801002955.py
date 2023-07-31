# Example to describe activity of a neuron 
# in a neural network
import math

# input signals
x1 = 0.7
x2 = 0.43

# weights of arrows
w1 = 3.2
w2 = 1.5

# bias to modify output independent of inputs
bias = -10

summation = w1*x1 + w2*x2 + bias
output = 1/(1+math.exp(-summation))

print(summation, " ", output)

122