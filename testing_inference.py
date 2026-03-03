import torch
from time import perf_counter
import os
import torch
from torchvision.transforms import v2
import timm
from PIL import Image
from transformers import CLIPModel


def make_transform_PIL(resize_size: int = 224):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


REPO_DIR = "."

# backbone = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224", num_classes=0, pretrained=True)
backbone = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").vision_model

backbone = backbone.to("cuda")

BATCH_SIZE = 50
IMG_ROOT = "../datasets/stripedmice/re-id/Full/"
images = []
for img in os.listdir(IMG_ROOT)[:BATCH_SIZE]:
    images.append(os.path.join(IMG_ROOT, img))

transform = make_transform_PIL()


data = torch.stack([transform(Image.open(img).convert("RGB")) for img in images])

times = []
for _ in range(50):
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            data = data.to("cuda")
            t0 = perf_counter()
            features = backbone(data)
            torch.cuda.synchronize()
            times.append(perf_counter() - t0)

mean_delay = torch.mean(torch.tensor(times[5:]))

print(f"Batch size: {BATCH_SIZE}, mean FPS: {1/mean_delay:.3f}")
