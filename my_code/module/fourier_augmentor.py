"""Fourier hard-negative image preprocessing component.

The component accepts two images before tensor conversion. It transfers the
anchor's centered low-frequency amplitude to the hard negative while retaining
the hard negative's phase, then returns a new RGB PIL image.
"""

import argparse
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


ImageInput = Union[Image.Image, np.ndarray, str, Path]


class FourierAugmentor:
    """Create a Fourier-augmented hard-negative image during preprocessing.

    Args:
        phi: Side-length proportion of the central low-frequency square. It
            must be in ``(0, 1]``.
        size_policy: ``"anchor-to-negative"`` resizes the anchor to the hard
            negative's size before augmentation. ``"error"`` rejects images
            with different dimensions.

    The callable interface accepts PIL RGB images, RGB NumPy arrays, or file
    paths and returns a PIL RGB image. Use ``augment_array`` when a NumPy RGB
    array is required instead.
    """

    def __init__(self, phi: float = 0.1, size_policy: str = "anchor-to-negative"):
        if not 0.0 < phi <= 1.0:
            raise ValueError("phi must be in the interval (0, 1]")
        if size_policy not in {"anchor-to-negative", "error"}:
            raise ValueError(
                "size_policy must be 'anchor-to-negative' or 'error'; got {!r}"
                .format(size_policy)
            )
        self.phi = float(phi)
        self.size_policy = size_policy

    def __repr__(self) -> str:
        return "{}(phi={}, size_policy={!r})".format(
            self.__class__.__name__, self.phi, self.size_policy
        )

    def __call__(
        self, anchor: ImageInput, hard_negative: ImageInput
    ) -> Image.Image:
        """Return a Fourier-augmented RGB PIL image."""
        anchor_image = self._as_rgb_image(anchor, "anchor")
        hard_negative_image = self._as_rgb_image(hard_negative, "hard_negative")
        anchor_image = self._align_anchor(anchor_image, hard_negative_image)
        output = self.augment_array(
            np.asarray(anchor_image), np.asarray(hard_negative_image)
        )
        return Image.fromarray(output, mode="RGB")

    def augment_array(
        self, anchor: np.ndarray, hard_negative: np.ndarray
    ) -> np.ndarray:
        """Return an RGB uint8 array from two equally sized RGB image arrays."""
        anchor_array = self._as_rgb_array(anchor, "anchor")
        hard_negative_array = self._as_rgb_array(hard_negative, "hard_negative")
        if anchor_array.shape != hard_negative_array.shape:
            raise ValueError(
                "anchor and hard_negative arrays must have equal shapes; got {} and {}"
                .format(anchor_array.shape, hard_negative_array.shape)
            )

        height, width, _ = anchor_array.shape
        mask = self._low_frequency_mask(height, width)[..., np.newaxis]
        anchor_spectrum = np.fft.fft2(anchor_array.astype(np.float32), axes=(0, 1))
        negative_spectrum = np.fft.fft2(
            hard_negative_array.astype(np.float32), axes=(0, 1)
        )
        anchor_shifted = np.fft.fftshift(anchor_spectrum, axes=(0, 1))
        negative_shifted = np.fft.fftshift(negative_spectrum, axes=(0, 1))
        mixed_amplitude = np.where(
            mask, np.abs(anchor_shifted), np.abs(negative_shifted)
        )
        mixed_spectrum = mixed_amplitude * np.exp(1j * np.angle(negative_shifted))
        output = np.fft.ifft2(
            np.fft.ifftshift(mixed_spectrum, axes=(0, 1)), axes=(0, 1)
        ).real
        return np.clip(np.rint(output), 0, 255).astype(np.uint8)

    def augment_to_file(
        self,
        anchor: ImageInput,
        hard_negative: ImageInput,
        output_path: Union[str, Path],
    ) -> Path:
        """Create and save an augmented RGB image, returning its output path."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self(anchor, hard_negative).save(output_path)
        return output_path

    def _low_frequency_mask(self, height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=bool)
        mask_height = max(1, int(round(height * self.phi)))
        mask_width = max(1, int(round(width * self.phi)))
        row_start = (height - mask_height) // 2
        column_start = (width - mask_width) // 2
        mask[
            row_start:row_start + mask_height,
            column_start:column_start + mask_width,
        ] = True
        return mask

    def _align_anchor(
        self, anchor: Image.Image, hard_negative: Image.Image
    ) -> Image.Image:
        if anchor.size == hard_negative.size:
            return anchor
        if self.size_policy == "error":
            raise ValueError(
                "Input image sizes differ: anchor is {}, hard negative is {}"
                .format(anchor.size, hard_negative.size)
            )
        return anchor.resize(hard_negative.size, Image.Resampling.BICUBIC)

    @classmethod
    def _as_rgb_image(cls, value: ImageInput, name: str) -> Image.Image:
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if isinstance(value, (str, Path)):
            path = Path(value)
            if not path.is_file():
                raise FileNotFoundError("{} image does not exist: {}".format(name, path))
            with Image.open(path) as image:
                return image.convert("RGB")
        if isinstance(value, np.ndarray):
            return Image.fromarray(cls._as_rgb_array(value, name), mode="RGB")
        raise TypeError(
            "{} must be a PIL image, RGB NumPy array, or image path; got {}"
            .format(name, type(value).__name__)
        )

    @staticmethod
    def _as_rgb_array(value: np.ndarray, name: str) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise TypeError("{} must be a NumPy array".format(name))
        if value.ndim != 3 or value.shape[2] != 3:
            raise ValueError(
                "{} must have RGB array shape [H, W, 3]; got {}"
                .format(name, value.shape)
            )
        if value.dtype == np.uint8:
            return value
        if not np.issubdtype(value.dtype, np.number):
            raise TypeError("{} array must use a numeric dtype".format(name))
        array = value.astype(np.float32, copy=False)
        if array.size and array.min() >= 0.0 and array.max() <= 1.0:
            array = array * 255.0
        return np.clip(np.rint(array), 0, 255).astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True, help="Anchor image supplying low-frequency amplitudes.")
    parser.add_argument("--hard-negative", type=Path, required=True, help="Hard-negative image supplying phase and high-frequency amplitudes.")
    parser.add_argument("--output", type=Path, required=True, help="Output RGB image path. Parent directories are created when needed.")
    parser.add_argument("--phi", type=float, default=0.1, help="Central low-frequency mask side-length proportion in (0, 1]. Default: %(default)s")
    parser.add_argument("--size-policy", choices=("anchor-to-negative", "error"), default="anchor-to-negative", help="Handling for different sizes: anchor-to-negative resizes the anchor; error stops. Default: %(default)s")
    return parser.parse_args()


def main():
    args = parse_args()
    augmenter = FourierAugmentor(
        phi=args.phi, size_policy=args.size_policy
    )
    output_path = augmenter.augment_to_file(
        args.anchor, args.hard_negative, args.output
    )
    with Image.open(args.hard_negative) as hard_negative:
        width, height = hard_negative.size
    print("Anchor: {}".format(args.anchor.resolve()))
    print("Hard negative: {}".format(args.hard_negative.resolve()))
    print("Output: {}".format(output_path.resolve()))
    print("Phi: {}".format(args.phi))
    print("Output size: {}x{}".format(width, height))


if __name__ == "__main__":
    main()
