import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

endpoint = "https://eastus.api.cognitive.microsoft.com/"
key = ""

text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))
documents = [
    """
    Microsoft was founded by Bill Gates with some friends he met at Harvard. One of his friends,
    Steve Ballmer, eventually became CEO after Bill Gates as well. Steve Ballmer eventually stepped
    down as CEO of Microsoft, and was succeeded by Satya Nadella.
    Microsoft originally moved its headquarters to Bellevue, Washington in January 1979, but is now
    headquartered in Redmond.
    """
]

result = text_analytics_client.recognize_linked_entities(documents)
docs = [doc for doc in result if not doc.is_error]

print(
    "Let's map each entity to it's Wikipedia article. I also want to see how many times each "
    "entity is mentioned in a document\n\n"
)
entity_to_url = {}
for doc in docs:
    for entity in doc.entities:
        print("Entity '{}' has been mentioned '{}' time(s)".format(
            entity.name, len(entity.matches)
        ))
        if entity.data_source == "Wikipedia":
            entity_to_url[entity.name] = entity.url

