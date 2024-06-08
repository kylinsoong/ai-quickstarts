import tensorflow as tf

data = tf.data.Dataset.from_tensor_slices({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
})

for record in data:
    print(record)


def preprocess(record):
    record['Age'] = record['Age'] + 1  # Example transformation
    return record

processed_data = data.map(preprocess)

for record in processed_data:
    print(record)
