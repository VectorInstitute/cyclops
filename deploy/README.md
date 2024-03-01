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
