import tensorflow as tf

def preprocess(line):
    return tf.strings.lower(line)

file_paths = ["data/data.txt", "data/data2.txt"]

dataset = tf.data.TextLineDataset(file_paths)

print(dataset)

for line in dataset:
    print(line.numpy().decode('utf-8'))

print()

dataset = dataset.map(preprocess)

for line in dataset:
    print(line.numpy().decode('utf-8'))

print()

dataset = dataset.shuffle(buffer_size=10).batch(2)

for batch in dataset:
    print(batch.numpy())
    for line in batch:
        print(line.numpy().decode('utf-8'))
