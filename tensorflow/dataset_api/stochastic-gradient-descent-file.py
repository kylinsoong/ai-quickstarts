import json
import math
import os
from pprint import pprint
import numpy as np
import tensorflow as tf

def create_dataset(pattern):
    return tf.data.experimental.make_csv_dataset(pattern, 1, CSV_COLUMNS, DEFAULTS)

def features_and_labels(row_data):
    label = row_data.pop(LABEL_COLUMN)
    features = row_data
    
    for unwanted_col in UNWANTED_COLS:
        features.pop(unwanted_col)

    return features, label

def create_dataset(pattern, batch_size):
    dataset = tf.data.experimental.make_csv_dataset(
        pattern, batch_size, CSV_COLUMNS, DEFAULTS)
    return dataset.map(features_and_labels)

def loss_mse(X, Y, w0, w1):
    Y_hat = w0 * X + w1
    errors = (Y_hat - Y)**2
    return tf.reduce_mean(errors)

def compute_gradients(X, Y, w0, w1):
    with tf.GradientTape() as tape:
        loss = loss_mse(X, Y, w0, w1)
    return tape.gradient(loss, [w0, w1])

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

tempds = create_dataset('./toy_data/taxi-train*')

print(tempds)

for data in tempds.take(2):
    pprint({k: v.numpy() for k, v in data.items()})
    print("\n")

UNWANTED_COLS = ['pickup_datetime', 'key']

for row_data in tempds.take(2):
    features, label = features_and_labels(row_data)
    pprint(features)
    print(label, "\n")
    assert UNWANTED_COLS[0] not in features.keys()
    assert UNWANTED_COLS[1] not in features.keys()
    assert label.shape == [1]

BATCH_SIZE = 2

tempds = create_dataset('./toy_data/taxi-train*', batch_size=2)

for X_batch, Y_batch in tempds.take(2):
    pprint({k: v.numpy() for k, v in X_batch.items()})
    print(Y_batch.numpy(), "\n")
    assert len(Y_batch) == BATCH_SIZE
