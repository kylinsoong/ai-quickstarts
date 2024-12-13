import random
import string
import numpy as np
import pandas as pd
from google.cloud import aiplatform, bigquery

PROJECT_ID = "888888888888"
REGION = "europe-west4"

aiplatform.init(project=PROJECT_ID, location=REGION)

def generate_uuid(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))

def download_bq_table(bq_table_uri: str) -> pd.DataFrame:
    prefix = "bq://"
    if bq_table_uri.startswith(prefix):
        bq_table_uri = bq_table_uri[len(prefix) :]

    table = bigquery.TableReference.from_string(bq_table_uri)

    bqclient = bigquery.Client(project=PROJECT_ID)

    rows = bqclient.list_rows(
        table,
    )
    return rows.to_dataframe()


UUID = generate_uuid()

print("UUID:", UUID)

BQ_SOURCE = "bq://bigquery-public-data.ml_datasets.penguins"
penguins_df = download_bq_table(BQ_SOURCE)
penguins_df.index = penguins_df.index.map(str)
NA_VALUES = ["NA", "."]
penguins_df = penguins_df.replace(to_replace=NA_VALUES, value=np.nan).dropna()

print(penguins_df.head(3))
print(penguins_df.info)

FEATURESTORE_ID = f"penguins_{UUID}"

penguins_feature_store = aiplatform.Featurestore.create(
    featurestore_id=FEATURESTORE_ID,
    online_store_fixed_node_count=1,
    project=PROJECT_ID,
    location=REGION,
    sync=True,
)

print("Creating Featurestore")

fs = aiplatform.Featurestore(
    featurestore_name=FEATURESTORE_ID,
    project=PROJECT_ID,
    location=REGION,
)
print("Featurestore:", fs.gca_resource)

ENTITY_TYPE_ID = f"penguin_entity_type_{UUID}"

# Create penguin entity type
penguins_entity_type = penguins_feature_store.create_entity_type(
    entity_type_id=ENTITY_TYPE_ID,
    description="Penguins entity type",
)

print("Creating entity type")

entity_type = penguins_feature_store.get_entity_type(entity_type_id=ENTITY_TYPE_ID)

print("Entity type:", entity_type.gca_resource)

penguins_feature_configs = {
    "species": {
        "value_type": "STRING",
    },
    "island": {
        "value_type": "STRING",
    },
    "culmen_length_mm": {
        "value_type": "DOUBLE",
    },
    "culmen_depth_mm": {
        "value_type": "DOUBLE",
    },
    "flipper_length_mm": {
        "value_type": "DOUBLE",
    },
    "body_mass_g": {"value_type": "DOUBLE"},
    "sex": {"value_type": "STRING"},
}

penguin_features = penguins_entity_type.batch_create_features(
    feature_configs=penguins_feature_configs,
)

print("Creating features")

penguins_entity_type.preview.write_feature_values(instances=penguins_df)

ENTITY_IDS = [str(x) for x in range(100)]
penguins_entity_type.read(entity_ids=ENTITY_IDS)

penguins_feature_store.delete(force=True)
