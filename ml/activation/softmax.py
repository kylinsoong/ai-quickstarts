import math

def softmax(x):
    e_x = [math.exp(i) for i in x]
    sum_e_x = sum(e_x)
    return [i / sum_e_x for i in e_x]

input_array = [-5.0, 0.0, 5.0]
output_array = softmax(input_array)
print("Softmax input: ", input_array)
print("Softmax Output:", output_array)
