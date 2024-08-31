import os
import typing
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient, HealthcareEntityRelation

endpoint = "https://eastus.api.cognitive.microsoft.com/"
key = ""

text_analytics_client = TextAnalyticsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key),
)

documents = [
    """
    Patient needs to take 100 mg of ibuprofen, and 3 mg of potassium. Also needs to take
    10 mg of Zocor.
    """,
    """
    Patient needs to take 50 mg of ibuprofen, and 2 mg of Coumadin.
    """
]

poller = text_analytics_client.begin_analyze_healthcare_entities(documents)
result = poller.result()

docs = [doc for doc in result if not doc.is_error]

print("Let's first visualize the outputted healthcare result:")
for doc in docs:
    for entity in doc.entities:
        print(f"Entity: {entity.text}")
        print(f"...Normalized Text: {entity.normalized_text}")
        print(f"...Category: {entity.category}")
        print(f"...Subcategory: {entity.subcategory}")
        print(f"...Offset: {entity.offset}")
        print(f"...Confidence score: {entity.confidence_score}")
        if entity.data_sources is not None:
            print("...Data Sources:")
            for data_source in entity.data_sources:
                print(f"......Entity ID: {data_source.entity_id}")
                print(f"......Name: {data_source.name}")
        if entity.assertion is not None:
            print("...Assertion:")
            print(f"......Conditionality: {entity.assertion.conditionality}")
            print(f"......Certainty: {entity.assertion.certainty}")
            print(f"......Association: {entity.assertion.association}")
    for relation in doc.entity_relations:
        print(f"Relation of type: {relation.relation_type} has the following roles")
        for role in relation.roles:
            print(f"...Role '{role.name}' with entity '{role.entity.text}'")
    print("------------------------------------------")

print("Now, let's get all of medication dosage relations from the documents")
dosage_of_medication_relations = [
    entity_relation
    for doc in docs
    for entity_relation in doc.entity_relations if entity_relation.relation_type == HealthcareEntityRelation.DOSAGE_OF_MEDICATION
]
