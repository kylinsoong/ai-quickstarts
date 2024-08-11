import tensorflow as tf
import numpy as np

input_data = np.array([[3, 5]], dtype=float)
dense_layer = tf.keras.layers.Dense(units=2, activation='relu', use_bias=True)
dense_layer.build(input_shape=(None, 2))

print("Input:\n", input_data)
print("Kernel (Weights):\n", dense_layer.kernel.numpy())
print("Bias:\n", dense_layer.bias.numpy())

output = dense_layer(input_data)

print("Output:\n", output.numpy())
