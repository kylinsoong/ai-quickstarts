import tensorflow as tf

def parse(record):
    feature_description = {
        'id': tf.io.FixedLenFeature([], tf.int64),
        'value': tf.io.FixedLenFeature([], tf.float32),
        'name': tf.io.FixedLenFeature([], tf.string)
    }
    return tf.io.parse_single_example(record, feature_description)

dataset = tf.data.TFRecordDataset('data/data.tfrecord')
dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.map(lambda record: parse(record))
dataset = dataset.batch(batch_size=2)

for element in dataset:
    print("id:", element['id'].numpy(), 
          ", name:", element['name'].numpy(), 
          ", value:", element['value'].numpy())

