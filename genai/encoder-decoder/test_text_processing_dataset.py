import tensorflow as tf
from fileutils import readAsString

text = readAsString("https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")
vocab = sorted(set(text))
ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
all_ids = ids_from_chars(tf.strings.unicode_split(text, "UTF-8"))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
seq_length = 100
sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

for e in dataset:
    print(e)

print("\n")
for e in dataset.take(1):
   print(e)
