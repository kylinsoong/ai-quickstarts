import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

endpoint = "https://eastus.api.cognitive.microsoft.com/"
key = ""

text_analytics_client = TextAnalyticsClient(
    endpoint=endpoint, credential=AzureKeyCredential(key)
)
documents = [
    """Parker Doe has repaid all of their loans as of 2020-04-25.
    Their SSN is 859-98-0987. To contact them, use their phone number
    555-555-5555. They are originally from Brazil and have Brazilian CPF number 998.214.865-68"""
]

result = text_analytics_client.recognize_pii_entities(documents)
docs = [doc for doc in result if not doc.is_error]

print(
    "Let's compare the original document with the documents after redaction. "
    "I also want to comb through all of the entities that got redacted"
)
for idx, doc in enumerate(docs):
    print(f"Document text: {documents[idx]}")
    print(f"Redacted document text: {doc.redacted_text}")
    for entity in doc.entities:
        print("...Entity '{}' with category '{}' got redacted".format(
            entity.text, entity.category
        ))
