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

get_product_info = FunctionDeclaration(
    name="get_product_info",
    description="Get the stock amount and identifier for a given product",
    parameters={
        "type": "object",
        "properties": {
            "product_name": {"type": "string", "description": "Product name"}
        },
    },
)

get_store_location = FunctionDeclaration(
    name="get_store_location",
    description="Get the location of the closest store",
    parameters={
        "type": "object",
        "properties": {"location": {"type": "string", "description": "Location"}},
    },
)

place_order = FunctionDeclaration(
    name="place_order",
    description="Place an order",
    parameters={
        "type": "object",
        "properties": {
            "product": {"type": "string", "description": "Product name"},
            "address": {"type": "string", "description": "Shipping address"},
        },
    },
)

retail_tool = Tool(
    function_declarations=[
        get_product_info,
        get_store_location,
        place_order,
    ],
)

model = GenerativeModel(
    "gemini-1.5-pro-001",
    generation_config=GenerationConfig(temperature=0),
    tools=[retail_tool],
)
chat = model.start_chat()

prompt = """
Do you have the Pixel 8 Pro in stock?
"""

response = chat.send_message(prompt)
#print(response.text)
print(response.candidates[0].content.parts[0])

api_response = {"sku": "GA04834-US", "in_stock": "yes"}
response = chat.send_message(
    Part.from_function_response(
        name="get_product_info",
        response={
            "content": api_response,
        },
    ),
)
print(response.text)

prompt = """
What about the Pixel 8? Is there a store in
Mountain View, CA that I can visit to try one out?
"""

response = chat.send_message(prompt)
print(response.candidates[0].content.parts[0])

api_response = {"sku": "GA08475-US", "in_stock": "yes"}
response = chat.send_message(
    Part.from_function_response(
        name="get_product_info",
        response={
            "content": api_response,
        },
    ),
)
print(response.candidates[0].content.parts[0])

api_response = {"store": "2000 N Shoreline Blvd, Mountain View, CA 94043, US"}
response = chat.send_message(
    Part.from_function_response(
        name="get_store_location",
        response={
            "content": api_response,
        },
    ),
)
print(response.text)

prompt = """
I'd like to order a Pixel 8 Pro and have it shipped to 1155 Borregas Ave, Sunnyvale, CA 94089.
"""

response = chat.send_message(prompt)
print(response.candidates[0].content.parts[0])

api_response = {
    "payment_status": "paid",
    "order_number": 12345,
    "est_arrival": "2 days",
}
response = chat.send_message(
    Part.from_function_response(
        name="place_order",
        response={
            "content": api_response,
        },
    ),
)
print(response.text)
