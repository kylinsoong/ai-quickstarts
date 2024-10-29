import tensorflow as tf
import os
from fileutils import modelDir

one_step = modelDir("one_step")
model = tf.saved_model.load(one_step)

states = None
next_char = tf.constant(["Generated sentence:\n\n"])
result = [next_char]

for n in range(150):
    next_char, states = model.generate_one_step(
        next_char, states=states
    )
    result.append(next_char)


print(tf.strings.join(result)[0].numpy().decode("utf-8"))
