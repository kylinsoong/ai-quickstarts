import math

def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

print(tanh(-5))
print(tanh(0))
print(tanh(5))
