
def leaky_relu(x):
    return max(0.01*x, x)

input_array = [-5.0, 0.0, 5.0]
output_array = [leaky_relu(-5), leaky_relu(0), leaky_relu(5)]

print("Leaky ReLU input: ", input_array)
print("Leaky ReLU output:", output_array)

