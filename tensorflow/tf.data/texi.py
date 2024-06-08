import os
from pprint import pprint
import tensorflow as tf

print(tf.version.VERSION)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CSV_COLUMNS = [
    'fare_amount',
    'pickup_datetime',
    'pickup_longitude',
    'pickup_latitude',
    'dropoff_longitude',
    'dropoff_latitude',
    'passenger_count',
    'key'
]
LABEL_COLUMN = 'fare_amount'

DEFAULTS = [[0.0], ['na'], [0.0], [0.0], [0.0], [0.0], [0.0], ['na']]

def create_dataset(pattern):
    return tf.data.experimental.make_csv_dataset(pattern, 1, CSV_COLUMNS, DEFAULTS)


tempds = create_dataset('../data/toy_data/taxi-train*')
print(tempds)

for data in tempds.take(2):
    pprint({k: v.numpy() for k, v in data.items()})
    print("\n")

UNWANTED_COLS = ['pickup_datetime', 'key']

def features_and_labels(row_data):
    label = row_data.pop(LABEL_COLUMN)
    features = row_data
    
    for unwanted_col in UNWANTED_COLS:
        features.pop(unwanted_col)

    return features, label

for row_data in tempds.take(2):
    features, label = features_and_labels(row_data)
    pprint(features)
    print(label, "\n")
    assert UNWANTED_COLS[0] not in features.keys()
    assert UNWANTED_COLS[1] not in features.keys()
    assert label.shape == [1]

def create_dataset(pattern, batch_size):
    dataset = tf.data.experimental.make_csv_dataset(
        pattern, batch_size, CSV_COLUMNS, DEFAULTS)
    return dataset.map(features_and_labels)

BATCH_SIZE = 2

tempds = create_dataset('../data/toy_data/taxi-train*', batch_size=2)

for X_batch, Y_batch in tempds.take(2):
    pprint({k: v.numpy() for k, v in X_batch.items()})
    print(Y_batch.numpy(), "\n")
    assert len(Y_batch) == BATCH_SIZE

def create_dataset(pattern, batch_size=1, mode="eval"):
    dataset = tf.data.experimental.make_csv_dataset(
        pattern, batch_size, CSV_COLUMNS, DEFAULTS)

    dataset = dataset.map(features_and_labels).cache()

    if mode == "train":
        dataset = dataset.shuffle(1000).repeat()

    dataset = dataset.prefetch(1)
    return dataset

tempds = create_dataset('../data/toy_data/taxi-train*', 2, "train")
print(list(tempds.take(1)))

tempds = create_dataset('../data/toy_data/taxi-valid*', 2, "eval")
print(list(tempds.take(1)))
