import pandas as pd
from lib import build_model,train_model,plot_the_model_sample,plot_the_loss_curve

def predict_house_values(training_df, n, feature, label):

  batch = training_df[feature][10000:10000 + n]
  predicted_values = my_model.predict_on_batch(x=batch)

  print("feature   label          predicted")
  print("  value   value          value")
  print("          in thousand$   in thousand$")
  print("--------------------------------------")
  for i in range(n):
    print ("%5.0f %6.0f %15.0f" % (training_df[feature][10000 + i],
                                   training_df[label][10000 + i],
                                   predicted_values[i][0] ))

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

training_df = pd.read_csv('data/california_housing_train.csv')
training_df["median_house_value"] /= 1000.0
training_df["rooms_per_person"] = training_df["total_rooms"] / training_df["population"]

print(training_df.head())
print(training_df.describe())
print(training_df.corr())

learning_rate = 0.06
epochs = 24
batch_size = 30

my_feature = "median_income"
my_label="median_house_value" 
my_model = build_model(learning_rate)
weight, bias, epochs, rmse = train_model(my_model, training_df[my_feature], 
                                         training_df[my_label],
                                         epochs, batch_size)

plot_the_model_sample(training_df, weight, bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)

predict_house_values(training_df, 10, my_feature, my_label)
