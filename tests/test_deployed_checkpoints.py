import os
import pytest
import yaml
import pandas as pd
import torchvision.transforms as T
import torch
import onnxruntime as ort
from addict import Dict
from timm import create_model

from wildlife_tools.fork_additions import (
    deploy,
)
from wildlife_tools.data import WildlifeDataset

ROOT = "../tests/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_onnx(session: ort.InferenceSession, input_tensor: torch.Tensor) -> torch.Tensor:
    """Run ONNX inference and return output as torch tensor."""
    input_np = input_tensor.cpu().numpy()
    output = session.run(None, {"input": input_np})[0]
    return torch.from_numpy(output)


def run_trt(context, input_tensor: torch.Tensor) -> torch.Tensor:
    """Run TensorRT inference and return output as torch tensor."""
    input_tensor = input_tensor.cuda().contiguous()
    output_tensor = torch.empty(input_tensor.shape[0], 768, dtype=input_tensor.dtype, device=DEVICE)
    context.set_input_shape("input", input_tensor.shape)
    context.set_tensor_address("input", input_tensor.data_ptr())
    context.set_tensor_address("output", output_tensor.data_ptr())
    context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return output_tensor


@pytest.fixture
def config():
    with open(os.path.join(ROOT, "../configs/user_configs.yaml"), "r") as f:
        config = Dict(yaml.safe_load(f))
    config["save_directory"] = ""
    config["dataset_directory"] = os.path.join(ROOT, "TestDataset")
    config["device"] = DEVICE
    return config


def test_deployed_checkpoints(config):
    # deploy(config)

    ckpt_dir = os.path.abspath(os.path.join(ROOT, "..", "checkpoints"))
    onnx_ckpt_path = os.path.abspath(os.path.join(ckpt_dir, "precision_track_re-identificator.onnx"))
    trt_ckpt_path = os.path.abspath(
        os.path.join(ckpt_dir, "precision_track_re-identificator_NVIDIAGeForceRTX3090_FP16.engine")
    )

    pytorch_model = create_model("hf-hub:BVRA/MegaDescriptor-T-224", num_classes=0, pretrained=True)
    pytorch_model = pytorch_model.to(DEVICE)
    pytorch_model = pytorch_model.eval()

    onnx_session = ort.InferenceSession(onnx_ckpt_path)

    trt_context = None
    if DEVICE == "cuda":
        import tensorrt as trt

        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(trt_ckpt_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
            trt_engine = runtime.deserialize_cuda_engine(f.read())
        trt_context = trt_engine.create_execution_context()

    test_dataset = WildlifeDataset(
        metadata=pd.read_csv(os.path.join(config["dataset_directory"], "metadata.csv")),
        root=config["dataset_directory"],
        transform=T.Compose(
            [
                T.Resize(size=(224, 224)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        ),
    )

    for i in range(len(test_dataset)):
        img = test_dataset[i][0].unsqueeze(0).to(torch.float16)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pytorch_output = pytorch_model(img.to(DEVICE))
            onnx_output = run_onnx(onnx_session, img)
            assert torch.allclose(pytorch_output.cpu().half(), onnx_output, atol=1e-1)
            if DEVICE == "cuda":
                trt_output = run_trt(trt_context, img)
                assert torch.allclose(pytorch_output.half(), trt_output, atol=1e-1)


if __name__ == "__main__":
    pytest.main(["-x", os.path.realpath(__file__), "-s"])
