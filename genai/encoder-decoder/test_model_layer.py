import tensorflow as tf
import numpy as np

inputs = np.random.random((32, 10, 8))
gru = tf.keras.layers.GRU(4, return_sequences=True, return_state=True)
whole_sequence_output, final_state = gru(inputs)
print(whole_sequence_output.shape)
print(final_state.shape)
