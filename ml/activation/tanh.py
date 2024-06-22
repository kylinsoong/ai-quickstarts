import math

def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

input_array = [-5.0, 0.0, 5.0]
output_array = [tanh(-5), tanh(0), tanh(5)]

print("Tanh input: ", input_array)
print("Tanh output:", output_array)

