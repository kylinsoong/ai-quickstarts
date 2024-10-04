PROJECT_ID = "" 
LOCATION = "us-central1" 

import vertexai
import requests
from vertexai.generative_models import (
    GenerationConfig,
    Image,
    Part,
)

vertexai.init(project=PROJECT_ID, location=LOCATION)

multimodal_model = GenerativeModel(model_name="gemini-1.5-flash")

prompt = """
What is shown in this video?
Where should I go to see it?
What are the top 5 places in the world that look like this?
"""

video_uri = f"gs://github-repo/img/gemini/multimodality_usecases_overview/mediterraneansea.mp4"
video = Part.from_uri(
    uri=video_uri,
    mime_type="video/mp4",
)
contents = [prompt, video]

responses = multimodal_model.generate_content(contents, stream=True)

print("-------Prompt--------")
print_multimodal_prompt(contents)

for response in responses:
    print(response.text, end="")

