from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import os

class colors:
    green = '\033[92m'
    blue = '\033[94m'
    red = '\033[31m'
    yellow = '\033[33m'
    reset = '\033[0m'

# Paste your endpoint and key below
endpoint = "https://eastus.api.cognitive.microsoft.com/"
key = ""
client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

domain = "landmarks"
#url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-python-sdk-samples/master/samples/vision/images/make_things_happen.jpg"
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Broadway_and_Times_Square_by_night.jpg/450px-Broadway_and_Times_Square_by_night.jpg"
language = "en"
max_descriptions = 3

analysis = client.describe_image(url, max_descriptions, language)

input("Press Enter to list results...\n")

for caption in analysis.captions:
    print(caption)
    print(caption.text)
    print(caption.confidence)
