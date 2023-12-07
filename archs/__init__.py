from .types import PyTorchModel
from .ditn import DITN
from .omnisr import OmniSR
from .rrdb import RRDBNet as ESRGAN
from .srvgg import SRVGGNetCompact as RealESRGANv2
from .dat import DAT
from .swinir import SwinIR


def load_model(state_dict) -> PyTorchModel:
    state_dict_keys = list(state_dict.keys())

    if "params_ema" in state_dict_keys:
        state_dict = state_dict["params_ema"]
    elif "params-ema" in state_dict_keys:
        state_dict = state_dict["params-ema"]
    elif "params" in state_dict_keys:
        state_dict = state_dict["params"]

    state_dict_keys = list(state_dict.keys())

    # for key in state_dict.keys():
    #     print(key, state_dict[key].shape)

    model = None
    if "UFONE.0.ITLs.0.attn.temperature" in state_dict_keys:
        model = DITN(state_dict)
    elif "residual_layer.0.residual_layer.0.layer.0.fn.0.weight" in state_dict_keys:
        model = OmniSR(state_dict)
    elif "body.0.weight" in state_dict_keys and "body.1.weight" in state_dict_keys:
        model = RealESRGANv2(state_dict)
    elif "layers.0.residual_group.blocks.0.norm1.weight" in state_dict_keys:
        model = SwinIR(state_dict)
    elif "layers.0.blocks.2.attn.attn_mask_0" in state_dict_keys:
        model = DAT(state_dict)
    else:
        try:
            model = ESRGAN(state_dict)
        except:
            # pylint: disable=raise-missing-from
            raise Exception("UNSUPPORTED_MODEL")

    model.load_state_dict(state_dict, strict=False)

    return model
