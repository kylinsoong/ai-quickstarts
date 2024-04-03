import tensorflow as tf

m = tf.constant([[3, 5, 7], [4, 6, 8]])
x = tf.reshape(m, [3,2])
y = tf.reshape(m, [1,6])
z = tf.reshape(m, [6,1])

print(m)
print(x)
print(y)
print(z)

