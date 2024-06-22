import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


input_array = [-5.0, 0.0, 5.0]
output_array = [sigmoid(-5), sigmoid(0), sigmoid(5)]

print("Sigmoid input: ", input_array)
print("Sigmoid output:", output_array)
