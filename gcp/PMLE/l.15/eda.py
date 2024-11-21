import os
import requests
import urllib.parse

def download_file(url, local_filename):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()  # Check for HTTP errors

        with open(local_filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    file.write(chunk)

    print(f"Downloaded file: {local_filename}")

def readAsString(url):
    parsed_url = urllib.parse.urlparse(url)
    path = parsed_url.path
    filename = path.split('/')[-1]
    parts = filename.split('.')
    pathname = parts[0]
    base_dir = os.path.join(os.path.expanduser("~"), ".ai", pathname)
    file_path = os.path.join(base_dir, filename)

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print("create directory", base_dir)

    if not os.path.isfile(file_path):
        download_file(url, file_path)

    content = open(file_path, "rb").read().decode(encoding="utf-8")
    return content

def checkpointDir(dir):
    return os.path.join(os.path.expanduser("~"), ".ai", dir)

def modelDir(dir):
    return os.path.join(os.path.expanduser("~"), ".ai", "model", dir)


readAsString("https://storage.googleapis.com/cloud-training/mlongcp/v3.0_MLonGC/toy_data/housing_pre-proc_toy.csv")

df_USAhousing = pd.read_csv('../data/explore/housing_pre-proc_toy.csv')

df_USAhousing.head()

df_USAhousing.isnull().sum()

df_stats = df_USAhousing.describe()
df_stats = df_stats.transpose()

df_USAhousing.info()

print ("Rows     : " ,df_USAhousing.shape[0])
print ("Columns  : " ,df_USAhousing.shape[1])
print ("\nFeatures : \n" ,df_USAhousing.columns.tolist())
print ("\nMissing values :  ", df_USAhousing.isnull().sum().values.sum())
print ("\nUnique values :  \n",df_USAhousing.nunique())

sns.displot(df_USAhousing['median_house_value'])

sns.set_style('whitegrid')
df_USAhousing['median_house_value'].hist(bins=30)
plt.xlabel('median_house_value')

x = df_USAhousing['median_income']
y = df_USAhousing['median_house_value']
plt.scatter(x, y)
plt.show()

sns.jointplot(x='median_income',y='median_house_value',data=df_USAhousing)

sns.countplot(x = 'ocean_proximity', data=df_USAhousing)

g = sns.FacetGrid(df_USAhousing, col="ocean_proximity")
g.map(plt.hist, "households");

g = sns.FacetGrid(df_USAhousing, col="ocean_proximity")
g.map(plt.hist, "median_income");


x = df_USAhousing['latitude']
y = df_USAhousing['longitude']

plt.scatter(x, y)
plt.show()
