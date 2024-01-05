# Safe unpickler to prevent arbitrary code execution
# https://github.com/chaiNNer-org/spandrel/blob/ff7f11467db20a4e6ccd25280e233c1295857f5c/src/spandrel/__helpers/unpickler.py
import pickle
from types import SimpleNamespace

safe_list = {
    ("collections", "OrderedDict"),
    ("typing", "OrderedDict"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch", "BFloat16Storage"),
    ("torch", "FloatStorage"),
    ("torch", "HalfStorage"),
    ("torch", "IntStorage"),
    ("torch", "LongStorage"),
    ("torch", "DoubleStorage"),
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        # Only allow required classes to load state dict
        if (module, name) not in safe_list:
            raise pickle.UnpicklingError(f"Global '{module}.{name}' is forbidden")
        return super().find_class(module, name)


RestrictedUnpickle = SimpleNamespace(
    Unpickler=RestrictedUnpickler,
    __name__="pickle",
    load=lambda *args, **kwargs: RestrictedUnpickler(*args, **kwargs).load(),
)