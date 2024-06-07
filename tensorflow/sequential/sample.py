import pandas as pd


training_df = pd.read_csv('data/sample.csv')

print(training_df.head())
print(training_df.describe())
