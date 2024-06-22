
def relu(x):
    return max(0, x)

input_array = [-5.0, 0.0, 5.0]
output_array = [relu(-5), relu(0), relu(5)]

print("ReLU input: ", input_array)
print("ReLU output:", output_array)

