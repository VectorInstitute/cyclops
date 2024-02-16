"""Model serving service with Triton Inference Server as backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import bentoml
import numpy as np
import torchxrayvision as xrv
from torchvision import transforms


if TYPE_CHECKING:
    from PIL.Image import Image


def get_transform(image_size: int) -> transforms.Compose:
    """Get image transformation for model inference."""
    return transforms.Compose(
        [
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(image_size),
        ],
    )


triton_runner = bentoml.triton.Runner(  # type: ignore
    "triton_runner",
    "src/model_repo",
    tritonserver_type="http",
    cli_args=[
        "--exit-on-error=true",  # exits if any error occurs during initialization
        "--http-restricted-api=model-repository:access-key=admin",  # restrict access to load/unload APIs
        "--model-control-mode=explicit",  # enable explicit model loading/unloading
        "--load-model=resnet50_res512_all",
    ],
)
svc = bentoml.Service("model-service", runners=[triton_runner])


@svc.api(  # type: ignore
    input=bentoml.io.Multipart(im=bentoml.io.Image(), model_name=bentoml.io.Text()),
    output=bentoml.io.JSON(),
)
async def classify_xray(im: Image, model_name: str) -> dict[str, float]:
    """Classify X-ray image using specified model."""
    img = np.asarray(im)
    img = xrv.datasets.normalize(
        img,
        img.max(),
        reshape=True,  # normalize image to [-1024, 1024]
    )

    model_repo_index = await triton_runner.get_model_repository_index()  # type: ignore
    available_models = [model["name"] for model in model_repo_index]
    if model_name not in available_models:
        raise bentoml.exceptions.InvalidArgument(
            f"Expected model name to be one of {available_models}, but got {model_name}",
        )

    img_size = 224
    if "resnet" in model_name:
        img_size = 512

    img = get_transform(img_size)(img)

    if len(img.shape) == 3:
        img = img[None]  # add batch dimension

    InferResult = await getattr(triton_runner, model_name).async_run(img)  # noqa: N806
    return dict(
        zip(xrv.datasets.default_pathologies, InferResult.as_numpy("OUTPUT__0")[0]),
    )


@svc.api(
    input=bentoml.io.NumpyNdarray(dtype="float32", shape=(-1, 21)),
    output=bentoml.io.NumpyNdarray(dtype="int64", shape=(-1,)),
)
async def predict_heart_failure(X: np.ndarray) -> np.ndarray:  # type: ignore
    """Run inference on heart failure prediction model."""
    InferResult = await triton_runner.heart_failure_prediction.async_run(  # type: ignore # noqa: N806
        X,
    )
    return InferResult.as_numpy("label")  # type: ignore[no-any-return]


# Triton Model management API
@svc.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())  # type: ignore
async def model_config(input_model: dict[Literal["model_name"], str]) -> dict[str, Any]:
    """Retrieve model configuration from Triton Inference Server."""
    return await triton_runner.get_model_config(input_model["model_name"])  # type: ignore


@svc.api(input=bentoml.io.Text(), output=bentoml.io.JSON())  # type: ignore
async def unload_model(input_model: str, ctx: bentoml.Context) -> dict[str, str]:
    """Unload a model from memory."""
    await triton_runner.unload_model(input_model, headers=ctx.request.headers)  # type: ignore # noqa: E501
    return {"unloaded": input_model}


@svc.api(input=bentoml.io.Text(), output=bentoml.io.JSON())  # type: ignore
async def load_model(input_model: str, ctx: bentoml.Context) -> dict[str, str]:
    """Load a model into memory."""
    await triton_runner.load_model(input_model, headers=ctx.request.headers)  # type: ignore
    return {"loaded": input_model}


@svc.api(input=bentoml.io.Text(), output=bentoml.io.JSON())  # type: ignore
async def list_models(_: str, ctx: bentoml.Context) -> list[str]:
    """Return a list of models available in the model repository."""
    return await triton_runner.get_model_repository_index(headers=ctx.request.headers)  # type: ignore
