"""Create a Fourier-augmented hard negative from an anchor and hard-negative image."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb_image(path):
    """Load an image as RGB uint8 data without changing its dimensions."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Image does not exist: {}".format(path))
    with Image.open(path) as image:
        return image.convert("RGB")


def low_frequency_mask(height, width, phi):
    """Create a centered rectangular mask whose side lengths are phi of the image size."""
    mask = np.zeros((height, width), dtype=bool)
    mask_height = max(1, int(round(height * phi)))
    mask_width = max(1, int(round(width * phi)))
    row_start = (height - mask_height) // 2
    column_start = (width - mask_width) // 2
    mask[
        row_start:row_start + mask_height,
        column_start:column_start + mask_width,
    ] = True
    return mask


def fourier_augment(anchor, hard_negative, phi):
    """Inject anchor low-frequency amplitudes into the hard negative's spectrum.

    Both inputs must be RGB arrays with the same height and width. The returned
    image retains the hard negative's phase and therefore its spatial semantics.
    """
    anchor = np.asarray(anchor, dtype=np.float32)
    hard_negative = np.asarray(hard_negative, dtype=np.float32)
    if anchor.shape != hard_negative.shape:
        raise ValueError(
            "Anchor and hard-negative arrays must have identical shapes; got {} and {}"
            .format(anchor.shape, hard_negative.shape)
        )
    if anchor.ndim != 3 or anchor.shape[2] != 3:
        raise ValueError("Expected RGB image arrays with shape (height, width, 3)")

    height, width, _ = anchor.shape
    mask = low_frequency_mask(height, width, phi)[..., np.newaxis]
    anchor_spectrum = np.fft.fft2(anchor, axes=(0, 1))
    negative_spectrum = np.fft.fft2(hard_negative, axes=(0, 1))

    anchor_amplitude = np.abs(np.fft.fftshift(anchor_spectrum, axes=(0, 1)))
    shifted_negative = np.fft.fftshift(negative_spectrum, axes=(0, 1))
    negative_amplitude = np.abs(shifted_negative)
    negative_phase = np.angle(shifted_negative)
    mixed_amplitude = np.where(mask, anchor_amplitude, negative_amplitude)

    mixed_spectrum = mixed_amplitude * np.exp(1j * negative_phase)
    reconstructed = np.fft.ifft2(
        np.fft.ifftshift(mixed_spectrum, axes=(0, 1)), axes=(0, 1)
    ).real
    return np.clip(np.rint(reconstructed), 0, 255).astype(np.uint8)


def prepare_anchor(anchor_image, hard_negative_image, size_policy):
    """Align the anchor to the output size while preserving hard-negative geometry."""
    if anchor_image.size == hard_negative_image.size:
        return anchor_image
    if size_policy == "error":
        raise ValueError(
            "Input image sizes differ: anchor is {}, hard negative is {}. "
            "Use --size-policy anchor-to-negative to resize the anchor."
            .format(anchor_image.size, hard_negative_image.size)
        )
    return anchor_image.resize(hard_negative_image.size, Image.Resampling.BICUBIC)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anchor",
        type=Path,
        required=True,
        help="Anchor image supplying low-frequency amplitude information.",
    )
    parser.add_argument(
        "--hard-negative",
        type=Path,
        required=True,
        help="Hard-negative image supplying phase and high-frequency amplitude information.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output RGB image path. Parent directories are created when needed.",
    )
    parser.add_argument(
        "--phi",
        type=float,
        default=0.1,
        help="Central low-frequency mask side-length proportion in (0, 1]. Default: %(default)s",
    )
    parser.add_argument(
        "--size-policy",
        choices=("anchor-to-negative", "error"),
        default="anchor-to-negative",
        help="Handling for different input sizes: anchor-to-negative resizes the anchor to the hard negative; error stops. Default: %(default)s",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.phi <= 1:
        raise ValueError("--phi must be in the interval (0, 1]")

    anchor_image = load_rgb_image(args.anchor)
    hard_negative_image = load_rgb_image(args.hard_negative)
    anchor_image = prepare_anchor(
        anchor_image, hard_negative_image, args.size_policy
    )
    augmented = fourier_augment(
        np.asarray(anchor_image), np.asarray(hard_negative_image), args.phi
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(augmented, mode="RGB").save(output_path)
    print("Anchor: {}".format(Path(args.anchor).resolve()))
    print("Hard negative: {}".format(Path(args.hard_negative).resolve()))
    print("Output: {}".format(output_path.resolve()))
    print("Phi: {}".format(args.phi))
    print("Output size: {}x{}".format(augmented.shape[1], augmented.shape[0]))


if __name__ == "__main__":
    main()
