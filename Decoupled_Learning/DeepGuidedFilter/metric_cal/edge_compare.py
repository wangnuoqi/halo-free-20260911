import argparse
import os

import cv2
import numpy as np

# 计算在保留边缘的mask下，用于矫正的边缘和非边缘部分的对比图。
# 观察在这种分割的情况下，我们的结果差在哪里，好在哪里
# 以及选用的方案是否有效

def load_mask(mask_path: str) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask image: {mask_path}")

    # Support both strict binary masks and masks stored as 0/255 grayscale.
    return (mask > 0).astype(np.uint8)


def load_color_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return image


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if image.shape[:2] != mask.shape:
        raise ValueError(
            f"Image size {image.shape[:2]} does not match mask size {mask.shape}"
        )

    mask_3ch = mask[:, :, None]
    return image * mask_3ch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_image(path: str, image: np.ndarray) -> None:
    if not cv2.imwrite(path, image):
        raise IOError(f"Failed to save image: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use a binary edge mask to extract and save edge and non-edge regions from two images."
    )
    parser.add_argument("--mask", required=True, help="Path to the 0-1 binary mask image.")
    parser.add_argument("--image1", required=True, help="Path to the first image.")
    parser.add_argument("--image2", required=True, help="Path to the second image.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory used to save the extracted images.",
    )
    parser.add_argument(
        "--prefix1",
        default="image1",
        help="Output filename prefix for the first image.",
    )
    parser.add_argument(
        "--prefix2",
        default="image2",
        help="Output filename prefix for the second image.",
    )
    parser.add_argument(
        "--save-inverse",
        action="store_true",
        help="Also save non-edge (inverse mask) images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mask = load_mask(args.mask)
    image1 = load_color_image(args.image1)
    image2 = load_color_image(args.image2)

    edge_image1 = apply_mask(image1, mask)
    edge_image2 = apply_mask(image2, mask)

    ensure_dir(args.output_dir)

    mask_save_path = os.path.join(args.output_dir, "mask_binary.png")
    edge1_save_path = os.path.join(args.output_dir, f"{args.prefix1}_edge.png")
    edge2_save_path = os.path.join(args.output_dir, f"{args.prefix2}_edge.png")

    save_image(mask_save_path, mask * 255)
    save_image(edge1_save_path, edge_image1)
    save_image(edge2_save_path, edge_image2)

    print(f"Saved binary mask to: {mask_save_path}")
    print(f"Saved first edge image to: {edge1_save_path}")
    print(f"Saved second edge image to: {edge2_save_path}")

    if args.save_inverse:
        inv_mask = (1 - mask).astype(np.uint8)
        non_edge_image1 = apply_mask(image1, inv_mask)
        non_edge_image2 = apply_mask(image2, inv_mask)

        inv_mask_save_path = os.path.join(args.output_dir, "mask_non_edge.png")
        non_edge1_save_path = os.path.join(args.output_dir, f"{args.prefix1}_non_edge.png")
        non_edge2_save_path = os.path.join(args.output_dir, f"{args.prefix2}_non_edge.png")

        #save_image(inv_mask_save_path, inv_mask * 255)
        save_image(non_edge1_save_path, non_edge_image1)
        save_image(non_edge2_save_path, non_edge_image2)

        print(f"Saved non-edge mask to: {inv_mask_save_path}")
        print(f"Saved first non-edge image to: {non_edge1_save_path}")
        print(f"Saved second non-edge image to: {non_edge2_save_path}")


if __name__ == "__main__":
    main()
