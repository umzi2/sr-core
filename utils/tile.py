from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

from utils.image import get_h_w_c

Size = Tuple[int, int]


class Split:
    pass


@dataclass(frozen=True)
class Padding:
    top: int
    right: int
    bottom: int
    left: int

    @staticmethod
    def all(value: int) -> "Padding":
        return Padding(value, value, value, value)

    @staticmethod
    def to(value: Padding | int) -> Padding:
        if isinstance(value, int):
            return Padding.all(value)
        return value

    @property
    def horizontal(self) -> int:
        return self.left + self.right

    @property
    def vertical(self) -> int:
        return self.top + self.bottom

    @property
    def empty(self) -> bool:
        return self.top == 0 and self.right == 0 and self.bottom == 0 and self.left == 0

    def scale(self, factor: int) -> Padding:
        return Padding(
            self.top * factor,
            self.right * factor,
            self.bottom * factor,
            self.left * factor,
        )

    def min(self, other: Padding | int) -> Padding:
        other = Padding.to(other)
        return Padding(
            min(self.top, other.top),
            min(self.right, other.right),
            min(self.bottom, other.bottom),
            min(self.left, other.left),
        )

    def remove_from(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = get_h_w_c(image)

        return image[
            self.top : (h - self.bottom),
            self.left : (w - self.right),
            ...,
        ]


@dataclass(frozen=True)
class Region:
    x: int
    y: int
    width: int
    height: int

    @property
    def size(self) -> Size:
        return self.width, self.height

    def scale(self, factor: int) -> Region:
        return Region(
            self.x * factor,
            self.y * factor,
            self.width * factor,
            self.height * factor,
        )

    def intersect(self, other: Region) -> Region:
        x = max(self.x, other.x)
        y = max(self.y, other.y)
        width = min(self.x + self.width, other.x + other.width) - x
        height = min(self.y + self.height, other.y + other.height) - y
        return Region(x, y, width, height)

    def add_padding(self, pad: Padding) -> Region:
        return Region(
            x=self.x - pad.left,
            y=self.y - pad.top,
            width=self.width + pad.horizontal,
            height=self.height + pad.vertical,
        )

    def remove_padding(self, pad: Padding) -> Region:
        return self.add_padding(pad.scale(-1))

    def child_padding(self, child: Region) -> Padding:
        """
        Returns the padding `p` such that `child.add_padding(p) == self`.
        """
        left = child.x - self.x
        top = child.y - self.y
        right = self.width - child.width - left
        bottom = self.height - child.height - top
        return Padding(top, right, bottom, left)

    def read_from(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = get_h_w_c(image)
        if (w, h) == self.size:
            return image

        return image[
            self.y : (self.y + self.height),
            self.x : (self.x + self.width),
            ...,
        ]

    def write_into(self, lhs: np.ndarray, rhs: np.ndarray):
        h, w, c = get_h_w_c(rhs)
        assert (w, h) == self.size
        assert c == get_h_w_c(lhs)[2]

        if c == 1:
            if lhs.ndim == 2 and rhs.ndim == 3:
                rhs = rhs[:, :, 0]
            if lhs.ndim == 3 and rhs.ndim == 2:
                rhs = np.expand_dims(rhs, axis=2)

        lhs[
            self.y : (self.y + self.height),
            self.x : (self.x + self.width),
            ...,
        ] = rhs


def split_tile_size(tile_size: Size) -> Size:
    w, h = tile_size
    assert w >= 16 and h >= 16
    return max(16, w // 2), max(16, h // 2)


def auto_split(
    img: np.ndarray,
    tile_max_size,
    upscale,
    overlap: int = 16,
) -> np.ndarray:
    h, w, c = get_h_w_c(img)

    img_region = Region(0, 0, w, h)

    max_tile_size = (tile_max_size, tile_max_size)
    #print(f"Auto split image ({w}x{h}px @ {c}) with initial tile size {max_tile_size}.")

    if w <= max_tile_size[0] and h <= max_tile_size[1]:
        upscale_result = upscale(img)
        if not isinstance(upscale_result, Split):
            return upscale_result

        max_tile_size = split_tile_size(max_tile_size)

        print(
            "Unable to upscale the whole image at once. Reduced tile size to"
            f" {max_tile_size}."
        )

    start_x = 0
    start_y = 0
    result: Optional[np.ndarray] = None
    scale: int = 0

    restart = True
    while restart:
        restart = False

        tile_count_x = math.ceil(w / max_tile_size[0])
        tile_count_y = math.ceil(h / max_tile_size[1])
        tile_size_x = math.ceil(w / tile_count_x)
        tile_size_y = math.ceil(h / tile_count_y)

        for y in range(0, tile_count_y):
            if restart:
                break
            if y < start_y:
                continue

            for x in range(0, tile_count_x):
                if y == start_y and x < start_x:
                    continue

                tile = Region(
                    x * tile_size_x, y * tile_size_y, tile_size_x, tile_size_y
                ).intersect(img_region)
                pad = img_region.child_padding(tile).min(overlap)
                padded_tile = tile.add_padding(pad)

                upscale_result = upscale(padded_tile.read_from(img))

                if isinstance(upscale_result, Split):
                    max_tile_size = split_tile_size(max_tile_size)

                    new_tile_count_x = math.ceil(w / max_tile_size[0])
                    new_tile_count_y = math.ceil(h / max_tile_size[1])
                    new_tile_size_x = math.ceil(w / new_tile_count_x)
                    new_tile_size_y = math.ceil(h / new_tile_count_y)
                    start_x = (x * tile_size_x) // new_tile_size_x
                    start_y = (y * tile_size_x) // new_tile_size_y

                    print(
                        f"Split occurred. New tile size is {max_tile_size}. Starting at"
                        f" {start_x},{start_y}."
                    )

                    restart = True
                    break

                up_h, up_w, _ = get_h_w_c(upscale_result)
                current_scale = up_h // padded_tile.height
                assert current_scale > 0
                assert padded_tile.height * current_scale == up_h
                assert padded_tile.width * current_scale == up_w

                if result is None:
                    scale = current_scale
                    result = np.zeros((h * scale, w * scale, c), dtype=np.float32)

                assert current_scale == scale

                upscale_result = pad.scale(scale).remove_from(upscale_result)

                tile.scale(scale).write_into(result, upscale_result)

    assert result is not None
    return result
