import tensorflow as tf

def create_example(data):
    feature = {
        'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[data['id']])),
        'value': tf.train.Feature(float_list=tf.train.FloatList(value=[data['value']])),
        'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data['name'].encode()]))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

data = [
    {'id': 1, 'value': 0.1, 'name': 'Alice'},
    {'id': 2, 'value': 0.2, 'name': 'Bob'},
    {'id': 3, 'value': 0.3, 'name': 'Charlie'},
    {'id': 4, 'value': 0.2, 'name': 'Kylin'},
    {'id': 5, 'value': 0.3, 'name': 'Alex'},
    {'id': 6, 'value': 0.1, 'name': 'Ada'},
    {'id': 7, 'value': 0.2, 'name': 'Coco'}
]

with tf.io.TFRecordWriter('data/data.tfrecord') as writer:
    for record in data:
        example = create_example(record)
        writer.write(example.SerializeToString())
