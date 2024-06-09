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
from .rgt import RGT
from .atd import ATD
from .camixersr import camixersr
from .plksr import PLKSR
from .realplksr import realplksr
from .seemore import seemore

def load_model(state_dict) -> PyTorchModel:
    unwrap_keys = ["state_dict", "params_ema", "params-ema", "params", "model", "net"]
    for key in unwrap_keys:
        if key in state_dict and isinstance(state_dict[key], dict):
            state_dict = state_dict[key]
            break


    state_dict_keys = list(state_dict.keys())
    model: PyTorchModel | None = None
    if "UFONE.0.ITLs.0.attn.temperature" in state_dict_keys:
        model = DITN(state_dict)
    elif "residual_layer.0.residual_layer.0.layer.0.fn.0.weight" in state_dict_keys:
        model = OmniSR(state_dict)
    elif "body.0.weight" in state_dict_keys and "body.1.weight" in state_dict_keys:
        model = RealESRGANv2(state_dict)
    elif "body.0.local_block.norm_1.weight" in state_dict_keys:
        model = seemore(state_dict)
    elif "layers.0.residual_group.blocks.0.norm1.weight" in state_dict_keys:
        model = SwinIR(state_dict)
    elif "layers.0.blocks.2.attn.attn_mask_0" in state_dict_keys:
        if "layers.0.blocks.0.gamma" in state_dict_keys:
            model = RGT(state_dict)
        else:
            model = DAT(state_dict)
    elif "block_1.c1_r.sk.weight" in state_dict_keys:
        model = SPAN(state_dict)
    elif (
        "conv_final.weight" in state_dict_keys
        or "unet1.conv1.conv.0.weight" in state_dict_keys
        or "unet1.conv_bottom.weight" in state_dict_keys
    ):
        model = cugan(state_dict)
    elif "head.weight" in state_dict_keys:
        model = camixersr(state_dict)
    elif 'to_feat.weight' in state_dict_keys:
        model = SAFMN(state_dict)
    elif 'feats.1.channe_mixer.0.weight' in state_dict_keys:
        print(0)
        model = PLKSR(state_dict)
    elif 'feats.1.channel_mixer.0.weight' in state_dict_keys and "feats.1.norm.weight" in state_dict_keys:
        print(1)
        model = realplksr(state_dict)
    elif 'relative_position_index_SA' in state_dict_keys:
        print(2)
        model = ATD(state_dict)


    else:
        try:
            model = ESRGAN(state_dict)
        except:
            # pylint: disable=raise-missing-from
            print(state_dict_keys)
            raise Exception("UNSUPPORTED_MODEL")

    model.load_state_dict(state_dict, strict=False)

    return model
