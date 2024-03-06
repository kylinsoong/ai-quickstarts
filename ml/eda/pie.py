import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Arial Unicode MS'

df = pd.read_csv('math_2023.csv')

sns.pairplot(df)
plt.show()
