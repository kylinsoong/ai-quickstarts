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


# Change the URL between the quotes below to run your own images!
image_to_analyze = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Broadway_and_Times_Square_by_night.jpg/450px-Broadway_and_Times_Square_by_night.jpg"


image_analysis = computervision_client.analyze_image(image_to_analyze,visual_features=[VisualFeatureTypes.description,VisualFeatureTypes.tags])

print("\n-----Image Description-----")
for caption in image_analysis.description.captions:
    print(f"{colors.green}Confidence: {colors.reset}" + str(caption.confidence))
    print(f"{colors.green}Description: {colors.reset}" + caption.text)
print("----------\n")


input("Press Enter to continue to image tags...\n")


print("-----Image Tags-----")
for tag in image_analysis.tags:
    print(f"{colors.green}Tag:{colors.reset} {tag.name:<15}", 
    f" {colors.yellow}Confidence:{colors.reset} {tag.confidence:<15}")
print("----------\n")


input("\nPress Enter to Exit...")
