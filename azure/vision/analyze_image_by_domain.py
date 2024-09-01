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
endpoint = ""
key = ""
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

domain = "landmarks"
url = "https://images.pexels.com/photos/338515/pexels-photo-338515.jpeg"
language = "en"

analysis = computervision_client.analyze_image_by_domain(domain, url, language)

input("Press Enter to list results...\n")

for landmark in analysis.result["landmarks"]:
    print(landmark)
    print(landmark["name"])
    print(landmark["confidence"])


