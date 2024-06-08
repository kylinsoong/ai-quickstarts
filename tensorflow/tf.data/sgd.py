import os
import tensorflow as tf
from lib import create_dataset, loss_mse, compute_gradients

print(tf.version.VERSION)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

N_POINTS = 10

X = tf.constant(range(N_POINTS), dtype=tf.float32)
Y = 2 * X + 10

print(X)
print(Y)

BATCH_SIZE = 2
EPOCH = 250
LEARNING_RATE = .02

MSG = "STEP {step} - loss: {loss}, w0: {w0}, w1: {w1}"

w0 = tf.Variable(0.0)
w1 = tf.Variable(0.0)

dataset = create_dataset(X, Y, epochs=EPOCH, batch_size=BATCH_SIZE)
#loss = 1

for step, (X_batch, Y_batch) in enumerate(dataset):
    dw0, dw1 = compute_gradients(X_batch, Y_batch, w0, w1)
    w0.assign_sub(dw0 * LEARNING_RATE)
    w1.assign_sub(dw1 * LEARNING_RATE)

    if step % 100 == 0:
        loss = loss_mse(X_batch, Y_batch, w0, w1)
        print(MSG.format(step=step, loss=loss, w0=w0.numpy(), w1=w1.numpy()))
        
assert loss < 0.0001
assert abs(w0 - 2) < 0.001
assert abs(w1 - 10) < 0.001
