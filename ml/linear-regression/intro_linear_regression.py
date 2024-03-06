import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


print('=' * 100)
print('| Load the Dataset')
print('=' * 100)

df_USAhousing = pd.read_csv('USA_Housing_toy.csv')

print(type(df_USAhousing))
print(df_USAhousing.head())
print(df_USAhousing.isnull().sum())
print(df_USAhousing.describe())
print(df_USAhousing.info())
print(df_USAhousing,5)

print('=' * 100)
print('| Exploratory Data Analysis (EDA)')
print('=' * 100)

sns.pairplot(df_USAhousing)
plt.savefig('pairplot.png')

sns.displot(df_USAhousing['Price'])
plt.savefig('price_distribution.png')

#sns.heatmap(df_USAhousing.corr())
#plt.savefig('heatmap.png')

print('=' * 100)
print('| Training a Linear Regression Model')
print('=' * 100)

X = df_USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
y = df_USAhousing['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

lm = LinearRegression()

lm.fit(X_train,y_train)

print(lm.intercept_)

coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
print(coeff_df)

print('=' * 100)
print('| Predictions from your Model')
print('=' * 100)

predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
plt.savefig('scatter.png')

sns.displot((y_test-predictions),bins=50)
plt.savefig('predictions.png')

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
