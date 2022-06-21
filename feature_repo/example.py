# This is an example feature definition file

from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, String

# Read data from parquet files. Parquet is convenient for local development mode. For
# production, you can use your favorite DWH, such as BigQuery. See Feast documentation
# for more info.
events_data = FileSource(
    path="/home/amritk/projects/cyclops/feature_repo/data/events.parquet",
    timestamp_field="event_timestamp",
)

# Define an entity for the driver. You can think of entity as a primary key used to
# fetch features.
encounter = Entity(
    name="encounter",
    join_keys=["encounter_id"],
    value_type=ValueType.INT64,
)

# Our parquet files contain sample data that includes a driver_id column, timestamps and
# three feature column. Here we define a Feature View that will allow us to serve this
# data to our model online.
events_view = FeatureView(
    name="events_timeline",
    entities=["encounter"],
    ttl=timedelta(days=1),
    schema=[
        Field(name="event_value", dtype=Float32),
        Field(name="event_name", dtype=String),
        Field(name="event_category", dtype=String),
    ],
    online=True,
    source=events_data,
    tags={},
)

events_timeline_fs = FeatureService(name="events_timeline", features=[events_view])
