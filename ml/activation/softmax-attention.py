import math

def softmax(x):
    e_x = [math.exp(i) for i in x]
    sum_e_x = sum(e_x)
    return [i / sum_e_x for i in e_x]

input_array = [13, 9, 9]
output_array = softmax(input_array)
print(output_array)

