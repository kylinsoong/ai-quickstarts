import tensorflow as tf

N_POINTS = 10
X = tf.constant(range(N_POINTS), dtype=tf.float32)
Y = 2 * X + 10

dataset = tf.data.Dataset.from_tensor_slices((X, Y))

print(dataset)
for element in dataset:
    print(element[0].numpy(), element[1].numpy())

EPOCH = 3
dataset = dataset.repeat(EPOCH)
print(dataset)
for element in dataset:
    print(element[0].numpy(), element[1].numpy())

BATCH_SIZE = 4
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
print(dataset)
for element in dataset:
    print(element[0].numpy(), element[1].numpy())
