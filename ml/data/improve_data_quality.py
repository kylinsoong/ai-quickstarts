import tensorflow as tf
import requests
import os
import pandas as pd  
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def download_file(url, local_filename):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()  # Check for HTTP errors

        with open(local_filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    file.write(chunk)

    print(f"Downloaded file: {local_filename}")

base_dir = os.path.join(os.path.expanduser("~"), ".ml/transport")
file_path = os.path.join(base_dir, "untidy_vehicle_data_toy.csv")

if not os.path.exists(base_dir):
    os.makedirs(base_dir)
    print("create directory", base_dir)

if not os.path.isfile(file_path):
    url = 'https://storage.googleapis.com/cloud-training/mlongcp/v3.0_MLonGC/toy_data/untidy_vehicle_data_toy.csv'
    download_file(url, file_path)

df_transport = pd.read_csv(file_path)

print(df_transport.head())

print(df_transport.info())

print(df_transport,5)

print(df_transport.describe())

grouped_data = df_transport.groupby(['Zip Code','Model Year','Fuel','Make','Light_Duty','Vehicles'])

print(df_transport.groupby('Fuel').first())

print(df_transport.isnull().sum())

print (df_transport['Date'])
print (df_transport['Date'].isnull())

print (df_transport['Make'])
print (df_transport['Make'].isnull())

print (df_transport['Model Year'])
print (df_transport['Model Year'].isnull())

print ("Rows     : " ,df_transport.shape[0])
print ("Columns  : " ,df_transport.shape[1])
print ("\nFeatures : \n" ,df_transport.columns.tolist())
print ("\nUnique values :  \n",df_transport.nunique())
print ("\nMissing values :  ", df_transport.isnull().sum().values.sum())

print(df_transport.tail())

print(df_transport.isnull().sum())

df_transport = df_transport.apply(lambda x:x.fillna(x.value_counts().index[0]))

print(df_transport.isnull().sum())

df_transport['Date'] =  pd.to_datetime(df_transport['Date'], format='%m/%d/%Y')
df_transport['year'] = df_transport['Date'].dt.year
df_transport['month'] = df_transport['Date'].dt.month
df_transport['day'] = df_transport['Date'].dt.day

print(df_transport.info())

grouped_data = df_transport.groupby(['Make'])
print(df_transport.groupby('month').first())
