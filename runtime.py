from prefect.run_configs import LocalRun
import prefect
from prefect import task, Flow


@task
def extract_latest():
    logger = prefect.context.get("logger")
    logger.info("Extracting data")

@task
def predict():
    logger = prefect.context.get("logger")
    logger.info("Running prediction")

with Flow("runtime") as flow:
    extract_latest()
    predict()

# Configure extra environment variables for this flow,
# and set a custom working directory
flow.run_config = LocalRun(
    env={"SOME_VAR": "VALUE"},
    working_dir="/path/to/working-directory"
)


flow.run()






