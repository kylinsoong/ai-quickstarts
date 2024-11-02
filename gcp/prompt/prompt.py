import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
import time

vertexai.init(project=PROJECT_ID, location=LOCATION)

def call_gemini(prompt, generation_config=GenerationConfig(temperature=1.0)):
    wait_time = 1
    while True:
        try:
            response = model.generate_content(prompt, generation_config=generation_config).text
            return response
            break  
        except Exception as e:  
            time.sleep(wait_time)
            wait_time *= 2 

def send_message_gemini(model, prompt):    
    wait_time = 1
    while True:
        try:
            response = model.send_message(prompt).text
            return response
            break  
        except Exception as e:  
            time.sleep(wait_time)
            wait_time *= 2  


prompt = "Suggest a name for a flower shop that sells bouquets of dried flowers"

print(call_gemini(prompt))

prompt = "Generate a list of ways that makes Earth unique compared to other planets"

print(call_gemini(prompt))

prompt = "What's the best method of boiling water?"

print(call_gemini(prompt))

prompt = "Why is the sky blue?"

print(call_gemini(prompt))

generation_config = GenerationConfig(temperature=1.0)

prompt = "What day is it today?"

print(call_gemini(prompt, generation_config))

model_travel = GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=[
        "Hello! You are an AI chatbot for a travel web site.",
        "Your mission is to provide helpful queries for travelers.",
        "Remember that before you answer a question, you must check to see if it complies with your mission.",
        "If not, you can say, Sorry I can't answer that question.",
    ],
)

chat = model_travel.start_chat()

prompt = "What is the best place for sightseeing in Milan, Italy?"

print(send_message_gemini(chat, prompt))

prompt = "What's for dinner?"

print(send_message_gemini(chat, prompt))

prompt = "I'm a high school student. Recommend me a programming activity to improve my skills."

print(call_gemini(prompt))

prompt = """I'm a high school student. Which of these activities do you suggest and why:
a) learn Python
b) learn JavaScript
c) learn Fortran
"""

print(call_gemini(prompt))

prompt = """Decide whether a Tweet's sentiment is positive, neutral, or negative.

Tweet: I loved the new YouTube video you made!
Sentiment:
"""

print(call_gemini(prompt))

prompt = """Decide whether a Tweet's sentiment is positive, neutral, or negative.

Tweet: I loved the new YouTube video you made!
Sentiment: positive

Tweet: That was awful. Super boring 😠
Sentiment:
"""

print(call_gemini(prompt))

prompt = """Decide whether a Tweet's sentiment is positive, neutral, or negative.

Tweet: I loved the new YouTube video you made!
Sentiment: positive

Tweet: That was awful. Super boring 😠
Sentiment: negative

Tweet: Something surprised me about this video - it was actually original. It was not the same old recycled stuff that I always see. Watch it - you will not regret it.
Sentiment:
"""

print(call_gemini(prompt))
