import tensorflow as tf

# Define a variable
x = tf.Variable(3.0)

# Define a function
def function(x):
    return x**2 + 2*x + 1

# Compute the gradient of the function with respect to x
with tf.GradientTape() as tape:
    y = function(x)

# Compute the gradient
gradient = tape.gradient(y, x)

print("Value of x:", x.numpy())
print("Value of the function at x:", function(x).numpy())
print("Gradient of the function at x:", gradient.numpy())

