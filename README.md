# SRCore
Upscaler that supports multiple Image Super-Resolution architectures.
Example of use with the Upscaler helper class:
```py
from utils.upscaler import Upscaler, UpscalerVideo

upscaler = Upscaler("./4x-esrgan-model.pth", "./inputFolder", "./outputFolder", 256, "png")
upscaler_video = UpscalerVideo("./4x-esrgan-model.pth", "./inputFolder", "./outputFolder", 256, "mp4", "libx264", "aac")

upscaler.run()
upscaler_video.run()

```

# Commits
Before commit, read [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/). The repository is held to this standard.
## File Naming
All files are naming in Snake Case, architecture class names are always the same as their names. Example:
```py
# bad:
class omnisr:
  ...

# bad:
class Omnisr:
  ...

# good:
class OmniSR:
  ...
```

# Credits
Repository created for that [Colab Notebook](https://colab.research.google.com/drive/166GftgPwl0pi77mswolxhdnDQJCN2uK2?usp=sharing)

Some of the code was taken from these repositories:
* [muslll/neosr](https://github.com/muslll/neosr)
* [chaiNNer-org/chaiNNer](https://github.com/chaiNNer-org/chaiNNer)
