import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel, Image, Part

PROJECT_ID = ""
LOCATION = ""
vertexai.init(project=PROJECT_ID, location=LOCATION)

multimodal_model = GenerativeModel("gemini-1.0-pro-vision")

file_path = "github-repo/img/gemini/multimodality_usecases_overview/pixel8.mp4"
video_uri = f"gs://{file_path}"
video_url = f"https://storage.googleapis.com/{file_path}"

video = Part.from_uri(video_uri, mime_type="video/mp4")
contents = [prompt, video]

responses = multimodal_model.generate_content(contents, stream=True)

for response in responses:
    print(response.text, end="")
