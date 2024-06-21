
def leaky_relu(x):
    return max(0.01*x, x)

print(leaky_relu(-5))
print(leaky_relu(0))
print(leaky_relu(5))
