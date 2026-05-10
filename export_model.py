"""Export a torchvision model to TorchScript (model.pt).

Default: MobileNetV2 with ImageNet weights, scripted via torch.jit.script.
Run once before building Docker images:

    python3 export_model.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def export(output_path: Path, classes_path: Path) -> None:
    weights = MobileNet_V2_Weights.IMAGENET1K_V2
    model = mobilenet_v2(weights=weights)
    model.eval()

    scripted = torch.jit.script(model)
    scripted.save(str(output_path))
    print(f"Saved TorchScript model -> {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

    classes_path.write_text(json.dumps(list(weights.meta["categories"]), ensure_ascii=False, indent=2))
    print(f"Saved ImageNet class list -> {classes_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("model.pt"))
    parser.add_argument("--classes", type=Path, default=Path("imagenet_classes.json"))
    args = parser.parse_args()
    export(args.output, args.classes)


if __name__ == "__main__":
    main()
