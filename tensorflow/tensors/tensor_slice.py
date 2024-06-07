import tensorflow as tf

m = tf.constant([[3, 5, 7], [4, 6, 8]])
x = m[:, 2]
y = m[1, :]

print(m)
print(x)
print(y)

