import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

print("Ran the import statements.")

train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")
train_df = train_df.reindex(np.random.permutation(train_df.index))

train_df_mean = train_df.mean()
train_df_std = train_df.std()
train_df_norm = (train_df - train_df_mean)/train_df_std
test_df_norm = (test_df - train_df_mean) / train_df_std

print(train_df_norm.head())

threshold = 265000 
train_df_norm["median_house_value_is_high"] = (train_df["median_house_value"] > threshold).astype(float)
test_df_norm["median_house_value_is_high"] = (test_df["median_house_value"] > threshold).astype(float)
train_df_norm["median_house_value_is_high"].head(8000)

print(train_df_norm.head())

inputs = {
  'median_income': tf.keras.Input(shape=(1,)),
  'total_rooms': tf.keras.Input(shape=(1,))
}

def create_model(my_inputs, my_learning_rate, METRICS):
  concatenated_inputs = tf.keras.layers.Concatenate()(my_inputs.values())
  dense = layers.Dense(units=1, name='dense_layer', activation=tf.sigmoid)
  dense_output = dense(concatenated_inputs)
  my_outputs = {
    'dense': dense_output,
  }
  model = tf.keras.Model(inputs=my_inputs, outputs=my_outputs)

  model.compile(optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate=my_learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=METRICS)
  return model

def train_model(model, dataset, epochs, label_name, batch_size=None, shuffle=True):
  features = {name:np.array(value) for name, value in dataset.items()}
  label = np.array(features.pop(label_name))
  history = model.fit(x=features, y=label, batch_size=batch_size, epochs=epochs, shuffle=shuffle)
  epochs = history.epoch
  hist = pd.DataFrame(history.history)
  return epochs, hist

print("Defined the create_model and train_model functions.")

def plot_curve(epochs, hist, list_of_metrics):

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Value")

  for m in list_of_metrics:
    x = hist[m]
    plt.plot(epochs[1:], x[1:], label=m)

  plt.legend()
  plt.show()

print("Defined the plot_curve function.")

learning_rate = 0.001
epochs = 20
batch_size = 100
label_name = "median_house_value_is_high"
classification_threshold = 0.52

METRICS = [
           tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=classification_threshold),
           tf.keras.metrics.Precision(thresholds=classification_threshold, name='precision'),
           tf.keras.metrics.Recall(thresholds=classification_threshold, name="recall"),
          ]

my_model = create_model(inputs, learning_rate, METRICS)

epochs, hist = train_model(my_model, train_df_norm, epochs, label_name, batch_size)

list_of_metrics_to_plot = ['accuracy', "precision", "recall"]
plot_curve(epochs, hist, list_of_metrics_to_plot)


features = {name:np.array(value) for name, value in test_df_norm.items()}
label = np.array(features.pop(label_name))

print(my_model.evaluate(x = features, y = label, batch_size=batch_size))
