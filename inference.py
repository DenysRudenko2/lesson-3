"""Top-3 ImageNet inference over a TorchScript MobileNetV2.

Examples:
    python3 inference.py path/to/cat.jpg
    python3 inference.py ./samples/            # iterate every image in a dir
    python3 inference.py img.jpg --model model.pt --classes imagenet_classes.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ImageNet normalization (matches MobileNet_V2_Weights.IMAGENET1K_V2 transforms).
_PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_classes(path: Path) -> list[str]:
    if not path.exists():
        sys.exit(f"Class list not found: {path}. Run export_model.py first.")
    return json.loads(path.read_text())


def load_model(path: Path) -> torch.jit.ScriptModule:
    if not path.exists():
        sys.exit(f"TorchScript model not found: {path}. Run export_model.py first.")
    model = torch.jit.load(str(path), map_location="cpu")
    model.eval()
    return model


def iter_images(target: Path) -> Iterable[Path]:
    if target.is_dir():
        files = sorted(p for p in target.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            sys.exit(f"No images found in {target} (looked for {sorted(IMAGE_SUFFIXES)}).")
        yield from files
    else:
        yield target


def predict(model: torch.jit.ScriptModule, image_path: Path, classes: list[str], k: int = 3) -> list[tuple[str, float]]:
    image = Image.open(image_path).convert("RGB")
    tensor = _PREPROCESS(image).unsqueeze(0)
    with torch.inference_mode():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
    top_probs, top_idx = torch.topk(probs, k)
    return [(classes[i], float(p)) for p, i in zip(top_probs.tolist(), top_idx.tolist())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to an image file or a directory of images.")
    parser.add_argument("--model", type=Path, default=Path("model.pt"))
    parser.add_argument("--classes", type=Path, default=Path("imagenet_classes.json"))
    parser.add_argument("-k", "--top-k", type=int, default=3)
    args = parser.parse_args()

    classes = load_classes(args.classes)
    model = load_model(args.model)

    for image_path in iter_images(args.input):
        results = predict(model, image_path, classes, k=args.top_k)
        print(f"\n{image_path.name}:")
        for rank, (label, score) in enumerate(results, start=1):
            print(f"  {rank}. {label:<30s} {score * 100:6.2f}%")


if __name__ == "__main__":
    main()
