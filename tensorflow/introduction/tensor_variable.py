import tensorflow as tf

x = tf.Variable(2.0, dtype=tf.float32, name="x")
y = tf.Variable(3.0, dtype=tf.float32, name="y")
z = tf.Variable(2.0, dtype=tf.float32, name="z")
print(x)
x.assign(45.8)
print(x)
x.assign_add(4)
print(x)
x.assign_sub(3)
print(x)
x.assign(tf.multiply(x, y))
print(x)
x.assign(tf.divide(x, z))
print(x)

