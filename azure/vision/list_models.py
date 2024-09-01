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

input("Press Enter to list models...\n")

models = computervision_client.list_models()

for x in models.models_property:
    print(x)

