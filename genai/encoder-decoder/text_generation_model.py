import tensorflow as tf
from fileutils import readAsString

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = self.embedding(inputs, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x

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

BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = (
    dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
)

embedding_dim = 256
rnn_units = 1024

model = MyModel(
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
)

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(
        example_batch_predictions.shape,
        "# (batch_size, sequence_length, vocab_size)",
    )

print(model.summary())
