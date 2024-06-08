import tensorflow as tf
from pprint import pprint

def features_and_labels(row_data):
    label = row_data.pop(LABEL_COLUMN)
    features = row_data
    
    for unwanted_col in UNWANTED_COLS:
        features.pop(unwanted_col)

    return features, label

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
dataset = tf.data.experimental.make_csv_dataset('../data/toy_data/taxi-train*', 1, CSV_COLUMNS, DEFAULTS)

print(dataset)

for data in dataset.take(2):
    pprint({k: v.numpy() for k, v in data.items()})
    print("\n")

UNWANTED_COLS = ['pickup_datetime', 'key']
for row_data in dataset.take(2):
    features, label = features_and_labels(row_data)
    pprint(features)
    print(label, "\n")

BATCH_SIZE = 2
dataset = tf.data.experimental.make_csv_dataset('../data/toy_data/taxi-train*', BATCH_SIZE, CSV_COLUMNS, DEFAULTS)
dataset = dataset.map(features_and_labels).cache()
dataset = dataset.shuffle(1000).repeat()
dataset = dataset.prefetch(1)
print(dataset)
print(list(dataset.take(1)))
