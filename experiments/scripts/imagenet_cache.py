"""Cache the ImageNet-1k validation probabilities, embeddings, and labels.

The script loads the ResNet-50 ``IMAGENET1K_V2`` weights from a local file, runs one
forward pass over the validation images, and saves float16 softmax probabilities
(n, 1000), float16 penultimate embeddings (n, 2048), and int64 labels. Class indices
follow the sorted-WNID order of torchvision's pretrained models.
"""

from __future__ import annotations

import argparse
from collections.abc import Sized
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torchvision.datasets import ImageFolder, ImageNet
from torchvision.models import ResNet50_Weights, resnet50


def _build_dataset(
    root: Path, transform: object
) -> torch.utils.data.Dataset[tuple[torch.Tensor, int]]:
    """Load the validation split with ImageNet, or with ImageFolder as a fallback."""
    try:
        return cast(
            torch.utils.data.Dataset[tuple[torch.Tensor, int]],
            ImageNet(str(root), split="val", transform=transform),
        )
    except (FileNotFoundError, RuntimeError) as err:
        print(f"[fallback] torchvision.ImageNet unavailable ({err}); ImageFolder")
        return cast(
            torch.utils.data.Dataset[tuple[torch.Tensor, int]],
            ImageFolder(str(root / "val"), transform=transform),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagenet-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    weights_meta = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval().to(device)

    features: list[torch.Tensor] = []

    def _hook(_module: object, _inp: object, out: torch.Tensor) -> None:
        features.append(torch.flatten(out, 1).half().cpu())

    model.avgpool.register_forward_hook(_hook)

    dataset = _build_dataset(args.imagenet_root, weights_meta.transforms())
    dataset_size = len(cast(Sized, dataset))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    probs_chunks = []
    label_chunks = []
    with torch.no_grad():
        for step, (images, labels) in enumerate(loader):
            logits = model(images.to(device, non_blocking=True))
            probs_chunks.append(torch.softmax(logits, dim=1).half().cpu())
            label_chunks.append(labels)
            if step % 20 == 0:
                done = (step + 1) * args.batch_size
                print(f"[{done}/{dataset_size}]", flush=True)

    probs = torch.cat(probs_chunks).numpy().astype(np.float16)
    embeddings = torch.cat(features).numpy().astype(np.float16)
    labels_arr = torch.cat(label_chunks).numpy().astype(np.int64)
    top1 = float(np.mean(np.argmax(probs, axis=1) == labels_arr))
    print(f"top-1 accuracy {top1:.4f} on {len(labels_arr)} images")
    if not 0.78 <= top1 <= 0.84:
        raise SystemExit(
            f"top-1 {top1:.4f} outside the expected IMAGENET1K_V2 range; "
            "check class ordering or weights",
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        probs=probs,
        embeddings=embeddings,
        labels=labels_arr,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
