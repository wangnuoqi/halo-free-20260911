"""Create shared-scale RGB absolute-error maps for sampled RTV outputs."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


MODELS = ("stage3", "official", "nafnet_patch")


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    samples = args.samples_dir.resolve()
    output = (args.output_dir or samples / "error_maps").resolve()
    output.mkdir(parents=True, exist_ok=True)
    stems = sorted(path.name.removesuffix("__input.png")
                   for path in samples.glob("*__input.png"))
    if not stems:
        raise SystemExit(f"No sampled input images in {samples}")

    errors = {}
    global_max = 0.0
    for stem in stems:
        target = read_rgb(samples / f"{stem}__target.png")
        errors[stem] = {}
        for model in MODELS:
            prediction = read_rgb(samples / f"{stem}__{model}.png")
            if prediction.shape != target.shape:
                raise ValueError(f"Shape mismatch for {stem}/{model}: {prediction.shape} vs {target.shape}")
            # Mean absolute RGB error at each pixel, in the [0, 1] range.
            error = np.abs(prediction - target).mean(axis=2)
            errors[stem][model] = error
            global_max = max(global_max, float(error.max()))

    if global_max <= 0.0:
        global_max = 1.0
    for stem in stems:
        for model in MODELS:
            error = errors[stem][model]
            figure, axis = plt.subplots(figsize=(5.4, 4.3), dpi=150)
            image = axis.imshow(error, cmap="magma", vmin=0.0, vmax=global_max)
            axis.set_title(f"{model} | mean absolute RGB error")
            axis.axis("off")
            colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            colorbar.set_label("absolute error (0-1)")
            figure.tight_layout(pad=0.3)
            figure.savefig(output / f"{stem}__{model}_error.png", bbox_inches="tight")
            plt.close(figure)

    metadata = {
        "samples": len(stems),
        "models": list(MODELS),
        "error_definition": "mean absolute RGB difference between model prediction and target, after uint8 PNG decoding, normalized to [0,1]",
        "color_scale": {"vmin": 0.0, "vmax": global_max, "scope": "all sampled images and all three models"},
        "target": "__target.png (fixed RTV operator target)",
    }
    (output / "README.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
