import os
import warnings
import time
import numpy as np
import tensorflow as tf
from fileutils import readAsString

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

text = readAsString("https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")
print(f"Length of texts: {len(text)} characters")
print("First 50 letter:", text[:50])

vocab = sorted(set(text))
print("unique characters:", f"{len(vocab)}")

ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None
)

chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None
)

def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

all_ids = ids_from_chars(tf.strings.unicode_split(text, "UTF-8"))
print("text to ids:", all_ids)
