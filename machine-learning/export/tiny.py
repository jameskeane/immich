import logging
import os
import platform
import subprocess

import open_clip
import torch
from tinynn.converter import TFLiteConverter


class Wrapper(torch.nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.model = open_clip.create_model(
            "ViT-B-32",
            pretrained="openai",
            precision="fp16" if device.type == "cuda" else "fp32",
            jit=False,
            require_pretrained=True,
            device=device,
        )

    def forward(self, input_tensor: torch.FloatTensor):
        embedding = self.model.encode_image(input_tensor.half() if self.device.type == "cuda" else input_tensor)
        return embedding.float()


def main():
    if platform.machine() not in ("x86_64", "AMD64"):
        raise RuntimeError(f"Can only run on x86_64 / AMD64, not {platform.machine()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logging.warning("No CUDA available, cannot create fp16 model! "
                        "proceeding to create a fp32 model (use only for testing)")
    
    model = Wrapper(device)
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    dummy_input = torch.rand((1, 3, 224, 224))
    dummy_input = dummy_input.to(device)

    dummy_out = model(dummy_input)
    print(dummy_out.dtype, dummy_out.device, dummy_out.shape)

    jit = torch.jit.trace(model, dummy_input)
    output_name = "output_tensor"
    list(jit.graph.outputs())[0].setDebugName(output_name)
    tflite_model_path = "tiny-clip.tflite"
    output_path = os.path.join("out", tflite_model_path)

    converter = TFLiteConverter(jit, dummy_input, output_path, nchw_transpose=True)
    # segfaults on ARM, must run on x86_64 / AMD64
    converter.convert()

    armnn_model_path = "tiny-clip.armnn"
    os.environ.LD_LIBRARY_PATH = "armnn"
    subprocess.run(
        [
            "./ArmnnConverter",
            "-f",
            "tflite-binary",
            "-m",
            tflite_model_path,
            "-i",
            "input_tensor",
            "-o",
            "output_name",
            "-p",
            armnn_model_path,
        ]
    )

if __name__ == "__main__":
    with torch.no_grad():
        main()
