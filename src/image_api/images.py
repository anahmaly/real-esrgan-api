from __future__ import annotations

import warnings
from dataclasses import dataclass
from io import BytesIO

from PIL import Image, UnidentifiedImageError


class InvalidImage(ValueError):
    pass


class ImageTooLarge(ValueError):
    pass


class InvalidWorkerImage(RuntimeError):
    pass


@dataclass(frozen=True)
class ImageInfo:
    width: int
    height: int
    mode: str


def validate_image(
    data: bytes,
    *,
    max_bytes: int,
    max_width: int,
    max_height: int,
    max_pixels: int,
    worker_output: bool = False,
) -> ImageInfo:
    invalid_type = InvalidWorkerImage if worker_output else InvalidImage
    if not data:
        raise invalid_type("image is empty")
    if len(data) > max_bytes:
        raise ImageTooLarge("encoded image exceeds configured limit")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                width, height = image.size
                if width < 1 or height < 1:
                    raise invalid_type("image dimensions are invalid")
                if width > max_width or height > max_height or width * height > max_pixels:
                    raise ImageTooLarge("image dimensions exceed configured limits")
                image.verify()
            with Image.open(BytesIO(data)) as image:
                image.load()
                return ImageInfo(width, height, image.mode)
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise ImageTooLarge("image dimensions exceed configured limits") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        if isinstance(exc, (ImageTooLarge, InvalidImage, InvalidWorkerImage)):
            raise
        raise invalid_type("bytes are not a valid image") from exc


def validate_png_output(
    data: bytes,
    *,
    expected_size: tuple[int, int] | None,
    required_mode: str | None,
    max_bytes: int,
    max_pixels: int,
) -> None:
    maximum_size = expected_size or (max_pixels, max_pixels)
    info = validate_image(
        data,
        max_bytes=max_bytes,
        max_width=maximum_size[0],
        max_height=maximum_size[1],
        max_pixels=max_pixels,
        worker_output=True,
    )
    try:
        with Image.open(BytesIO(data)) as image:
            image_format = image.format
    except Exception as exc:
        raise InvalidWorkerImage("worker output is invalid") from exc
    if image_format != "PNG" or (
        expected_size is not None and (info.width, info.height) != expected_size
    ):
        raise InvalidWorkerImage("worker output contract mismatch")
    if required_mode is not None and info.mode != required_mode:
        raise InvalidWorkerImage("worker output mode mismatch")
