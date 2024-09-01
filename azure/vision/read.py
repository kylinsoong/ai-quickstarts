from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
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
client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

url = "https://github.com/Azure-Samples/cognitive-services-python-sdk-samples/raw/master/samples/vision/images/make_things_happen.jpg"
raw = True
numberOfCharsInOperationId = 36

rawHttpResponse = client.read(url, language="en", raw=True)

operationLocation = rawHttpResponse.headers["Operation-Location"]
idLocation = len(operationLocation) - numberOfCharsInOperationId
operationId = operationLocation[idLocation:]

result = client.get_read_result(operationId)

input("Press Enter to list results...\n")

if result.status == OperationStatusCodes.succeeded:

    for line in result.analyze_result.read_results[0].lines:
        print(line)
        print(line.text)
        print(line.bounding_box)
