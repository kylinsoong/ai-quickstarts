import math

def softmax(x):
    e_x = [math.exp(i) for i in x]
    sum_e_x = sum(e_x)
    return [i / sum_e_x for i in e_x]

input_array = [1.0, 2.0, 3.0]
output_array = softmax(input_array)
print("Softmax Output:", output_array)
