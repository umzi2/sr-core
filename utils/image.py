import numpy as np
import cv2
import os
import random
import string
import torch
import math
from torchvision.utils import make_grid
from typing import Tuple, List
from .file import split_file_path, get_ext, get_opencv_formats

MAX_VALUES_BY_DTYPE = {
    np.dtype("int8").name: 127,
    np.dtype("uint8").name: 255,
    np.dtype("int16").name: 32767,
    np.dtype("uint16").name: 65535,
    np.dtype("int32").name: 2147483647,
    np.dtype("uint32").name: 4294967295,
    np.dtype("int64").name: 9223372036854775807,
    np.dtype("uint64").name: 18446744073709551615,
    np.dtype("float32").name: 1.0,
    np.dtype("float64").name: 1.0,
}


def as_3d(img: np.ndarray) -> np.ndarray:
    """Given a grayscale image, this returns an image with 3 dimensions (image.ndim == 3)."""
    if img.ndim == 2:
        return np.expand_dims(img.copy(), axis=2)
    return img


def as_4d(img: np.ndarray) -> np.ndarray:
    """Given a grayscale image, this returns an image with 4 dimensions (image.ndim == 4)."""
    if img.ndim == 3:
        return np.expand_dims(img.copy(), axis=2)
    return img


def get_h_w_c(image: np.ndarray) -> Tuple[int, int, int]:
    """Returns the height, width, and number of channels."""
    h, w = image.shape[:2]
    c = 1 if image.ndim == 2 else image.shape[2]
    return h, w, c


def read_cv(path: str) -> np.ndarray | None:
    if get_ext(path) not in get_opencv_formats():
        return None

    img = None
    try:
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    except Exception as e:
        raise RuntimeError(
            f'Error reading image image from path "{path}". Image may be corrupt'
        ) from e

    return img


def cv_save_image(path: str, img: np.ndarray, params: List[int]):
    """
    A light wrapper around `cv2.imwrite` to support non-ASCII paths.
    """

    # Write image with opencv if path is ascii, since imwrite doesn't support unicode
    # This saves us from having to keep the image buffer in memory, if possible
    if path.isascii():
        cv2.imwrite(path, img, params)
    else:
        dirname, _, extension = split_file_path(path)
        try:
            temp_filename = f'temp-{"".join(random.choices(string.ascii_letters, k=16))}.{extension}'
            full_temp_path = os.path.join(dirname, temp_filename)
            cv2.imwrite(full_temp_path, img, params)
            os.rename(full_temp_path, path)
        except:
            _, buf_img = cv2.imencode(f".{extension}", img, params)
            with open(path, "wb") as outf:
                outf.write(buf_img)  # type: ignore


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == "float64":
                img = img.astype("float32")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (
        torch.is_tensor(tensor)
        or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))
    ):
        raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(
                _tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False
            ).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(
                "Only support 4D, 3D or 2D tensor. But received with dimension:"
                f" {n_dim}"
            )
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result
