from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
import os

class colors:
    green = '\033[92m'
    blue = '\033[94m'
    red = '\033[31m'
    yellow = '\033[33m'
    reset = '\033[0m'

# Paste your endpoint and key below
cog_endpoint = "https://westus.api.cognitive.microsoft.com/"
cog_key = ""
face_client = FaceClient(cog_endpoint, CognitiveServicesCredentials(cog_key))


# Change the URL between the quotes below to run your own faces through the Azure Face Service!
face_to_analyze = "https://raw.githubusercontent.com/pluralsight-cloud/AI-900-Artificial-Intelligence-Workloads-and-Considerations/main/images/image-analysis/2-azureface-couple.jpg"

detected_faces = face_client.face.detect_with_url(url=face_to_analyze, 
return_face_attributes=list(['headPose','glasses','accessories','blur','noise','occlusion']), 
return_face_landmarks=True,
return_face_id=False)
if not detected_faces:
    print("No faces were found in the image! Please try another image!")
    input("Press Enter to exit")
    quit()

print("\n-----Facial Attributes-----")
for count, face in enumerate(detected_faces):
    print(f"{colors.green}\nPerson Number: {colors.reset}" + str(count))
    for acc in face.face_attributes.accessories:
        print(f"{colors.red}Accessories: {colors.reset}" + str(acc))
    print(f"{colors.yellow}Glasses: {colors.reset}" + str(face.face_attributes.glasses))
    print(f"{colors.blue}Blur: {colors.reset}" + str(face.face_attributes.blur.blur_level))
    print(f"{colors.green}Occlusion: {colors.reset}" + str(face.face_attributes.occlusion))
    print(f"{colors.red}Noise Level: {colors.reset}" + str(face.face_attributes.noise.noise_level))
