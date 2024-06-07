import numpy as np
import pandas as pd

my_column_names = ['Eleanor', 'Chidi', 'Tahani', 'Jason']
my_data = np.random.randint(low=0, high=101, size=(3, 4))
df = pd.DataFrame(data=my_data, columns=my_column_names)

print(df)

print("\nSecond row of the Eleanor column: %d\n" % df['Eleanor'][1])

df['Janet'] = df['Tahani'] + df['Jason']

print(df)

print("Experiment with a reference:")
reference_to_df = df
print("  Starting value of df: %d" % df['Jason'][1])
print("  Starting value of reference_to_df: %d\n" % reference_to_df['Jason'][1])

df.at[1, 'Jason'] = df['Jason'][1] + 5
print("  Updated df: %d" % df['Jason'][1])
print("  Updated reference_to_df: %d\n\n" % reference_to_df['Jason'][1])

print("Experiment with a true copy:")
copy_of_my_dataframe = df.copy()

print("  Starting value of my_dataframe: %d" % df['Jason'][1])
print("  Starting value of copy_of_my_dataframe: %d\n" % copy_of_my_dataframe['Jason'][1])

df.at[1, 'Jason'] = df['Jason'][1] + 3
print("  Updated my_dataframe: %d" % df['Jason'][1])
print("  copy_of_my_dataframe does not get updated: %d" % copy_of_my_dataframe['Jason'][1])
