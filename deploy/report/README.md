# Model Report Deployment using FastAPI

1. Install the required dependencies with poetry.
```bash
poetry install --with deploy_report
```

The end points you can access are:

`/evaluate`: Lists all models that can be queried with the server and their status.
```bash
curl -X 'POST' \
  'http://cyclops.cluster.local:3000/list_models' \
  -H 'accept: application/json' \
  -H 'Content-Type: text/plain' \
  -H 'access-key: admin'
```

`/load_model`: load a given model to memory in preparation for inference.
```bash
curl -X 'POST' \
  'http://cyclops.cluster.local:3000/load_model' \
  -H 'accept: application/json' \
  -H 'Content-Type: text/plain' \
  -H 'access-key: admin' \
  -d 'densenet121_res224_all'
```
