from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image
import io
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

width = 50
height = 50
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Broadway_and_Times_Square_by_night.jpg/450px-Broadway_and_Times_Square_by_night.jpg"

thumbnail = client.generate_thumbnail(width, height, url)

input("Press Enter to list results...\n")

for x in thumbnail:
    image = Image.open(io.BytesIO(x))

image.save('thumbnail.jpg')
