import tensorflow as tf

def compute_gradients(X, Y, w0, w1):
  with tf.GradientTape() as tape:
    loss = loss_mse(X, Y, wo, w1)
    return tape.gradient(loss, [w0, w1])

we = tf.Variable(0.0)
w1 = tf.Variable(0.0)

dwe, dw1 = compute_gradients(X, Y, w0, w1)
