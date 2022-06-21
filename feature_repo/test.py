"""Test example of using the feature store."""

from pprint import pprint
from datetime import datetime, timedelta
import pandas as pd

from feast import FeatureStore


store = FeatureStore(repo_path=".")
feature_view = store.get_feature_view("events_timeline")
print(feature_view.features)
entities = store.list_entities()


# # The entity dataframe is the dataframe we want to enrich with feature values
# entity_df = pd.DataFrame.from_dict(
#     {
#         # entity's join key -> entity values
#         "encounter_id": [27523218, 20136936, 25780062],
#
#         # "event_timestamp" (reserved key) -> timestamps
#         "event_timestamp": [
#             datetime.now() - timedelta(minutes=11),
#             datetime.now() - timedelta(minutes=36),
#             datetime.now() - timedelta(minutes=73),
#         ],
#     }
# )
#
#
# training_df = store.get_historical_features(
#     entity_df=entity_df,
#     features=[
#         "events_timeline:event_value",
#         "events_timeline:event_name",
#         "events_timeline:event_category",
#     ],
# ).to_df()
#
# print("----- Feature schema -----\n")
# print(training_df.info())
#
# print()
# print("----- Example features -----\n")
# print(training_df.head())
#
#
# feature_vector = store.get_online_features(
#     features=[
#         "events_timeline:event_value",
#         "events_timeline:event_name",
#         "events_timeline:event_category",
#     ],
#     entity_rows=[
#         {"encounter_id": 27523218},
#         {"encounter_id": 28123938},
#     ],
# ).to_dict()
#
#
# print("----- Online features -----\n")
# pprint(feature_vector)


feature_service = store.get_feature_service("events_timeline")
features = store.get_online_features(
    features=feature_service,
    entity_rows=[
        {"encounter_id": 27523218},
    ],
).to_dict()
print(features)
