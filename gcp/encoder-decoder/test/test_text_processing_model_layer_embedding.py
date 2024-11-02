import tensorflow as tf
import numpy as np

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(1000, 64, input_length=10))
model.compile('rmsprop', 'mse')
input_array = np.random.randint(1000, size=(32, 10))
print(input_array)
output_array = model.predict(input_array)
print(output_array)

