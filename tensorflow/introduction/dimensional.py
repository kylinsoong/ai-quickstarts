import tensorflow as tf

x = tf.constant(3)
print(x)

y = tf.constant([3, 5, 7])
print(y)

z = tf.constant([[3, 5, 7], [4, 6, 8]])
print(z)

o = tf.constant([[[3, 5, 7], [4, 6, 8]],
                 [[1, 2, 3], [4, 5, 5]]])
print(o)

x1 = tf.constant([3, 5, 7])
x2 = tf.stack([x1, x1])
x3 = tf.stack([x2, x2, x2, x2])
x4 = tf.stack([x3, x3])
print(x1)
print(x2)
print(x3)
print(x4)
