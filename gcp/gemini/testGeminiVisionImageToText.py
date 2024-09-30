import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel, Image, Part

PROJECT_ID = ""
LOCATION = "us-west1"
vertexai.init(project=PROJECT_ID, location=LOCATION)

multimodal_model = GenerativeModel("gemini-1.0-pro-vision")

image = Image.load_from_file("image.jpg")

prompt = "Describe this image?"
contents = [image, prompt]

responses = multimodal_model.generate_content(contents, stream=True)
for response in responses:
    print(response.text, end="")
