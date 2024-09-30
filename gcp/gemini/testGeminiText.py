import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel, Image, Part

PROJECT_ID = ""
LOCATION = ""
vertexai.init(project=PROJECT_ID, location=LOCATION)

model = GenerativeModel("gemini-1.0-pro")

responses = model.generate_content("What is Arduino", stream=True)

for response in responses:
    print(response.text, end="")
