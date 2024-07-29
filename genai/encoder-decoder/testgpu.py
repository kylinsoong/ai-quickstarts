import tensorflow as tf

print("tensorflow version:", tf.__version__)

gpus = len(tf.config.list_physical_devices('gpu'))

print("Number of GPUs available:", gpus)
