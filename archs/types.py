from typing import Union

from .span import SPAN
from .ditn import DITN
from .omnisr import OmniSR
from .rrdb import RRDBNet as ESRGAN
from .srvgg import SRVGGNetCompact as RealESRGANv2
from .dat import DAT
from .swinir import SwinIR

PyTorchModel = Union[DITN, OmniSR, ESRGAN, RealESRGANv2, DAT, SwinIR, SPAN]
