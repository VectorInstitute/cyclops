# Model Deployment with BentoML and Triton Inference Server

1. Install the required dependencies with poetry.
```bash
poetry install --with deploy
```
2. Serialize trained model and move it to the `model_repo` directory. Then create
   a `config.pbtxt` file for the model.

   **Example - torchxrayvision model**
   ```python
   import torch
   import torchxrayvision as xrv


   model = xrv.models.ResNet(weights="resnet50-res512-all").eval().cuda()

   dummy_input = (-1024 - 1024) * torch.rand(1, 1, 512, 512) + 1024
   dummy_input = dummy_input.cuda()

   torch.jit.trace(model, dummy_input).save("model_repo/resnet50_res512_all/1/model.pt")
   ```
   See `model_repo/resnet50_res512_all/config.pbtxt` for an example of a pytorch model configuration file.

   **Example - sklearn model**
   ```python
   from skl2onnx import to_onnx


   onnx_model = to_onnx(
      <sklearn_model>,
      <input_data>,
      options={"zipmap": False},
   )
   with open("model_repo/<model_name>/1/model.onnx", "wb") as f:
      f.write(onnx_model.SerializeToString())
   ```
   See `model_repo/heart_failure_prediction/config.pbtxt` for an example of an ONNX model configuration file.
3. Create a service with BentoML with a triton runner. See `service.py` for an example.
4. Define a bentofile to specify which files to include in the bento. See `bentofile.yaml` for an example.
5. Build a bento.
```bash
bentoml build --do-not-track
```
6. Containerize the bento.
```bash
bentoml containerize -t model-service:alpha --enable-features=triton --do-not-track model-service:latest
```

7. Run the container with docker.
```bash
docker run -d --gpus=1 --rm -p 3000:3000 model-service:alpha
```

<details>
<summary><b>Access the endpoints from the Vector Cluster</b></summary>

If you have access to the the Vector cluster, the model service is running at
```
http://cyclops.cluster.local:3000
```

Here are some of the end points you can access and an example of how to do so:

`/list_models`: Lists all models that can be queried with the server and their status.
```bash
curl -X 'POST' \
  'http://cyclops.cluster.local:3000/list_models' \
  -H 'accept: application/json' \
  -H 'Content-Type: text/plain' \
  -H 'access-key: admin' \
  -d 'None'
```

`/load_model`: load a given model to memory in preparation for inference.
```bash
curl -X 'POST' \
  'http://cyclops.cluster.local:3000/load_model' \
  -H 'accept: application/json' \
  -H 'Content-Type: text/plain' \
  -H 'access-key: admin' \
  -d '"heart_failure_prediction"'
```

`/unload_model`: unload a model from memory, making it unavailable for inference.
```bash
curl -X 'POST' \
  'http://cyclops.cluster.local:3000/unload_model' \
  -H 'accept: application/json' \
  -H 'Content-Type: text/plain' \
  -H 'access-key: admin' \
  -d '"densenet121_res224_all"'
```

`/model_config`: returns the configurations for a given model.
```bash
curl -X 'POST' \
  'http://cyclops.cluster.local:3000/model_config' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"model_name": "resnet50_res512_all"}'
```

`/predict_heart_failure`: this is the inference endpoint for the heart failure prediction model.
```bash
curl -X 'POST' \
  'http://cyclops.cluster.local:3000/predict_heart_failure' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '[
  [
    0.36734694,
    0.45107794,
    0.80985915,
    0.56097561,
    0.7,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    1.0,
    0.0,
    0.0,
    1.0,
    0.0
  ]
]'
```

`/classify_xray`: this is the inference end point for the torchxrayvision models. The image and model name needs to be specified.
```bash
wget https://raw.githubusercontent.com/mlmed/torchxrayvision/master/tests/covid-19-pneumonia-58-prior.jpg && \
curl -X 'POST' \
  'http://cyclops.cluster.local:3000/classify_xray' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'im=@covid-19-pneumonia-58-prior.jpg;type=image/jpeg' \
  -F 'model_name=resnet50_res512_all'
```

In most cases, the only thing you need to change is the -d or -F option. Notice that
the `load_model`, `unload_model` and `list_model` endpoints require an access key. This is
to demonstrate the access control feature of the server.
