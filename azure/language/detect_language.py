import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

endpoint = "https://eastus.api.cognitive.microsoft.com/"
key = ""

text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))
documents = [
    """
    The concierge Paulette was extremely helpful. Sadly when we arrived the elevator was broken, but with Paulette's help we barely noticed this inconvenience.
    She arranged for our baggage to be brought up to our room with no extra charge and gave us a free meal to refurbish all of the calories we lost from
    walking up the stairs :). Can't say enough good things about my experience!
    """,
    """
    最近由于工作压力太大，我们决定去富酒店度假。那儿的温泉实在太舒服了，我跟我丈夫都完全恢复了工作前的青春精神！加油！
    """
]

result = text_analytics_client.detect_language(documents)
reviewed_docs = [doc for doc in result if not doc.is_error]

print("Let's see what language each review is in!")

for idx, doc in enumerate(reviewed_docs):
    print("Review #{} is in '{}', which has ISO639-1 name '{}'\n".format(
        idx, doc.primary_language.name, doc.primary_language.iso6391_name
    ))
