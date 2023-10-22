from typing import Union
from .ditn import DITN
from .omnisr import OmniSR

PyTorchModel = Union[
    DITN,
    OmniSR,
]