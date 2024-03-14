import tensorflow as tf

print(tf.__version__)

x = tf.constant([
  [3, 5, 7],
  [4, 6, 8]
])

y = x[:,1]
print(x)
print(y)

z = tf.reshape(x, [3,2])
print(z)

a = tf.Variable(2.0, dtype=tf.float32, name="my_ariable")

print(a)

a.assign(45.8)

print(a)

a.assign_add(4)

print(a)

a.assign_sub(3)

print(a)

