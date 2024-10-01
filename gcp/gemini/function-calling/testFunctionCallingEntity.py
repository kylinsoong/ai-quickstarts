PROJECT_ID = ""  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

import vertexai
import requests
from vertexai.generative_models import (
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Part,
    Tool,
)

vertexai.init(project=PROJECT_ID, location=LOCATION)

extract_log_data = FunctionDeclaration(
    name="extract_log_data",
    description="Extract details from error messages in raw log data",
    parameters={
        "type": "object",
        "properties": {
            "locations": {
                "type": "array",
                "description": "Errors",
                "items": {
                    "description": "Details of the error",
                    "type": "object",
                    "properties": {
                        "error_message": {
                            "type": "string",
                            "description": "Full error message",
                        },
                        "error_code": {"type": "string", "description": "Error code"},
                        "error_type": {"type": "string", "description": "Error type"},
                    },
                },
            }
        },
    },
)

extraction_tool = Tool(
    function_declarations=[extract_log_data],
)

model = GenerativeModel(
    "gemini-1.5-pro-001",
    generation_config=GenerationConfig(temperature=0),
    tools=[extraction_tool],
)

prompt = """
[15:43:28] ERROR: Could not process image upload: Unsupported file format. (Error Code: 308)
[15:44:10] INFO: Search index updated successfully.
[15:45:02] ERROR: Service dependency unavailable (payment gateway). Retrying... (Error Code: 5522)
[15:45:33] ERROR: Application crashed due to out-of-memory exception. (Error Code: 9001)
"""

response = model.generate_content(
    prompt,
    generation_config=GenerationConfig(temperature=0),
)

print(response.candidates[0].content.parts[0].function_call)
