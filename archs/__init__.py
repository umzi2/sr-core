from .span import SPAN
from .types import PyTorchModel
from .ditn import DITN
from .omnisr import OmniSR
from .rrdb import RRDBNet as ESRGAN
from .srvgg import SRVGGNetCompact as RealESRGANv2
from .dat import DAT
from .swinir import SwinIR
from .realcugan import cugan
from .safmn import SAFMN


def load_model(state_dict) -> PyTorchModel:
    unwrap_keys = ["state_dict", "params_ema", "params-ema", "params", "model", "net"]
    for key in unwrap_keys:
        if key in state_dict and isinstance(state_dict[key], dict):
            state_dict = state_dict[key]
            break

    state_dict_keys = list(state_dict.keys())
    model: PyTorchModel | None = None
    try:
        cugan3x = state_dict["unet1.conv_bottom.weight"].shape[2]
    except:
        cugan3x = 0
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
    elif "block_1.c1_r.sk.weight" in state_dict_keys:
        model = SPAN(state_dict)
    elif 'conv_final.weight' in state_dict_keys or 'unet1.conv1.conv.0.weight' in state_dict_keys or cugan3x == 5:
        model = cugan(state_dict)
    elif 'to_feat.weight' in state_dict_keys:
        model = SAFMN(state_dict)
    else:
        try:
            model = ESRGAN(state_dict)
        except:
            # pylint: disable=raise-missing-from
            print(state_dict_keys)
            raise Exception("UNSUPPORTED_MODEL")

    model.load_state_dict(state_dict, strict=False)

    return model
