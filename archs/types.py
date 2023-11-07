from typing import Union
from .ditn import DITN
from .omnisr import OmniSR
from .RRDB import RRDBNet as ESRGAN
from .SRVGG import SRVGGNetCompact as RealESRGANv2
from .DAT import DAT
from .SwinIR import SwinIR
PyTorchModel = Union[
    DITN,
    OmniSR,
    ESRGAN,
    RealESRGANv2,
    DAT,
    SwinIR
]