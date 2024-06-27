# Model Report Deployment using FastAPI

1. Install the required dependencies with poetry.

```bash
poetry install --with deploy_report
```

2. Run the evaluation server.

```bash
uvicorn api.main:app --reload --host <host_ip> --port <port>
```

The end points you can access are:

`/evaluate`: You can send model predictions, labels and metadata
for evaluation to this API endpoint using a POST request. For example:
```bash
curl -X POST 'http://<host_ip>:<port>/evaluate' \
  -H 'Content-Type: application/json' \
  -d '{
    "preds_prob": [0.2, 0.7, 0.1, 0.9],
    "target": [0, 1, 0, 1],
    "metadata": {
      "Age": [25, 30, 45, 50],
      "Sex": ["M", "F", "M", "F"]
    }
  }'
```

If this is the first evaluation for the model, a new model report is created.
Else, the previous model report is used to add the new evaluation.

The server serves the latest model report, and hence to view it you
can navigate to the server IP address either directly at `http://<host_ip>:<port>` or `http://<host_ip>:<port>/evaluate`. In the latter case, we essentially send a GET request to the ``evaluate`` API endpoint which fetches the latest model report.
