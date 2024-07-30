import tensorflow as tf

embedding_layer = tf.keras.layers.Embedding(63, 5)
input = tf.constant([1, 2, 3])
print(input)
result = embedding_layer(input)
print(result)

input = tf.constant([[0, 1, 2], [3, 4, 5]])
print(input)
result = embedding_layer(input)
print(result)


