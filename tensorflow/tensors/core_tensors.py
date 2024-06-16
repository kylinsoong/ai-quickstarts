import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

print(tf.__version__)

x = tf.constant([2, 3, 4])

print(x)

x = tf.Variable(2.0, dtype=tf.float32, name='my_variable')

print(x)

x.assign(45.8)

print(x)

x.assign_add(4)

print(x)

x.assign_sub(3)

print(x)

a = tf.constant([5, 3, 8]) 
b = tf.constant([3, -1, 2])
c = tf.add(a, b)
d = a + b

print(a, b, c, d)
print("c:", c)
print("d:", d)

a = tf.constant([5, 3, 8]) 
b = tf.constant([3, -1, 2])
c = tf.multiply(a, b)
d = a * b

print("c:", c)
print("d:", d)

a = tf.constant([5, 3, 8], dtype=tf.float32)
b = tf.math.exp(a)

print("b:", b)

a_py = [1, 2] 
b_py = [3, 4] 

tf.add(a_py, b_py)

a_np = np.array([1, 2])
b_np = np.array([3, 4])

tf.add(a_np, b_np)

a_tf = tf.constant([1, 2])
b_tf = tf.constant([3, 4])

tf.add(a_tf, b_tf) 

a_tf.numpy()
