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



