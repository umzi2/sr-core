from .types import PyTorchModel
from .ditn import DITN
from .omnisr import OmniSR


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
    else:
        raise Exception("UNSUPPORTED_MODEL")

    model.load_state_dict(state_dict, strict=False)

    return model
