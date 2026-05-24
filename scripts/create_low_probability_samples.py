#!/usr/bin/env python3
"""Create distorted "low-probability" image samples from ground-truth assets.

The goal is not to create arbitrary noise, but images that are still visibly
related to the originals while lying far from the natural image manifold.
This makes them useful as teaching examples of samples with low probability
under p_gt.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Distort images from assets/gt into a parallel folder of "
            '"low-probability" examples.'
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("assets/gt"),
        help="Root folder containing ground-truth images.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/low_probability_samples"),
        help="Root folder where distorted images will be written.",
    )
    parser.add_argument(
        "--variants-per-image",
        type=int,
        default=1,
        help="How many distorted variants to create for each input image.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducible distortions.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on how many input images to process.",
    )
    return parser.parse_args()


def collect_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    )


def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def random_affine(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    angle = rng.uniform(-18.0, 18.0)
    shear_x = rng.uniform(-0.22, 0.22)
    shear_y = rng.uniform(-0.12, 0.12)
    scale_x = rng.uniform(0.82, 1.18)
    scale_y = rng.uniform(0.82, 1.18)
    translate_x = rng.uniform(-0.08 * width, 0.08 * width)
    translate_y = rng.uniform(-0.08 * height, 0.08 * height)

    theta = math.radians(angle)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    a = scale_x * cos_t + shear_x * sin_t
    b = -sin_t + shear_x * cos_t
    d = sin_t + shear_y * cos_t
    e = scale_y * cos_t - shear_y * sin_t
    c = translate_x
    f = translate_y

    return image.transform(
        image.size,
        Image.Transform.AFFINE,
        (a, b, c, d, e, f),
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0),
    )


def pixelate(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    factor = rng.randint(4, 10)
    down_w = max(8, width // factor)
    down_h = max(8, height // factor)
    small = image.resize((down_w, down_h), Image.Resampling.BILINEAR)
    return small.resize((width, height), Image.Resampling.NEAREST)


def jpeg_artifacts(image: Image.Image, rng: random.Random) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=rng.randint(8, 24), optimize=False)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def add_gaussian_noise(image: Image.Image, rng: random.Random) -> Image.Image:
    array = np.asarray(image).astype(np.float32)
    sigma = rng.uniform(18.0, 42.0)
    noise = np.random.default_rng(rng.randint(0, 10_000_000)).normal(
        loc=0.0,
        scale=sigma,
        size=array.shape,
    )
    noisy = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="RGB")


def color_shuffle(image: Image.Image, rng: random.Random) -> Image.Image:
    channels = list(image.split())
    rng.shuffle(channels)
    shifted = Image.merge("RGB", channels)
    shifted = ImageEnhance.Color(shifted).enhance(rng.uniform(0.0, 2.5))
    shifted = ImageEnhance.Contrast(shifted).enhance(rng.uniform(0.5, 1.9))
    return shifted


def occlude_blocks(image: Image.Image, rng: random.Random) -> Image.Image:
    array = np.array(image, copy=True)
    height, width, _ = array.shape
    n_blocks = rng.randint(2, 6)
    for _ in range(n_blocks):
        block_w = rng.randint(max(8, width // 10), max(12, width // 3))
        block_h = rng.randint(max(8, height // 10), max(12, height // 3))
        x0 = rng.randint(0, max(0, width - block_w))
        y0 = rng.randint(0, max(0, height - block_h))
        if rng.random() < 0.5:
            fill = np.array(
                [rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)],
                dtype=np.uint8,
            )
            array[y0 : y0 + block_h, x0 : x0 + block_w] = fill
        else:
            patch = array[y0 : y0 + block_h, x0 : x0 + block_w]
            array[y0 : y0 + block_h, x0 : x0 + block_w] = patch[::-1, ::-1]
    return Image.fromarray(array, mode="RGB")


def frequency_like_stripes(image: Image.Image, rng: random.Random) -> Image.Image:
    array = np.asarray(image).astype(np.float32)
    height, width, _ = array.shape
    xs = np.arange(width, dtype=np.float32)[None, :]
    ys = np.arange(height, dtype=np.float32)[:, None]
    wave = (
        25.0 * np.sin(xs / rng.uniform(3.0, 11.0))
        + 18.0 * np.cos(ys / rng.uniform(4.0, 13.0))
    )
    channel_scale = np.array(
        [rng.uniform(0.6, 1.4), rng.uniform(0.6, 1.4), rng.uniform(0.6, 1.4)],
        dtype=np.float32,
    )
    striped = np.clip(array + wave[..., None] * channel_scale, 0, 255).astype(np.uint8)
    return Image.fromarray(striped, mode="RGB")


def weird_tone_map(image: Image.Image, rng: random.Random) -> Image.Image:
    transformed = ImageOps.posterize(image, bits=rng.randint(2, 5))
    if rng.random() < 0.7:
        transformed = ImageOps.solarize(transformed, threshold=rng.randint(40, 180))
    if rng.random() < 0.35:
        transformed = ImageOps.invert(transformed)
    transformed = ImageEnhance.Sharpness(transformed).enhance(rng.uniform(0.0, 3.0))
    return transformed


def blur_then_over_sharpen(image: Image.Image, rng: random.Random) -> Image.Image:
    blurred = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.5, 4.5)))
    return ImageEnhance.Sharpness(blurred).enhance(rng.uniform(2.5, 6.0))


DISTORTIONS = [
    ("affine", random_affine),
    ("pixelate", pixelate),
    ("jpeg", jpeg_artifacts),
    ("noise", add_gaussian_noise),
    ("channel_shuffle", color_shuffle),
    ("occlusion", occlude_blocks),
    ("stripes", frequency_like_stripes),
    ("tone_map", weird_tone_map),
    ("blur_sharpen", blur_then_over_sharpen),
]


def distort_image(image: Image.Image, rng: random.Random) -> tuple[Image.Image, list[str]]:
    image = ensure_rgb(image)
    names_and_ops = rng.sample(DISTORTIONS, k=rng.randint(3, 5))
    applied: list[str] = []
    distorted = image
    for name, op in names_and_ops:
        distorted = op(distorted, rng)
        applied.append(name)
    return distorted, applied


def output_path_for(
    image_path: Path,
    input_root: Path,
    output_root: Path,
    variant_idx: int,
) -> Path:
    relative = image_path.relative_to(input_root)
    stem = relative.stem
    suffix = relative.suffix.lower() if relative.suffix.lower() in VALID_SUFFIXES else ".png"
    target_dir = output_root / relative.parent
    return target_dir / f"{stem}_lowprob_{variant_idx:02d}{suffix}"


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    images = collect_images(args.input_root)
    if args.max_images is not None:
        images = images[: args.max_images]

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []

    for image_path in images:
        original = Image.open(image_path)
        for variant_idx in range(args.variants_per_image):
            distorted, applied = distort_image(original, rng)
            out_path = output_path_for(
                image_path=image_path,
                input_root=args.input_root,
                output_root=args.output_root,
                variant_idx=variant_idx,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            distorted.save(out_path)
            manifest.append(
                {
                    "source": str(image_path),
                    "output": str(out_path),
                    "distortions": applied,
                }
            )

    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"Created {len(manifest)} distorted images under {args.output_root} "
        f"from {len(images)} source images."
    )
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
