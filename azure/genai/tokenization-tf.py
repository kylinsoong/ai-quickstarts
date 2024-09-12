
import tensorflow as tf

text = ["I heard a dog bark loudly at a cat"]

vectorizer = tf.keras.layers.TextVectorization(output_mode='int')

vectorizer.adapt(text)
 
print(vectorizer("I heard a dog bark loudly at a cat").numpy())
print(vectorizer("I heard a cat").numpy())
