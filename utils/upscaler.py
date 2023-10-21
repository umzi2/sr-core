import os
import numpy as np
import torch
from archs import load_model
from utils.cuda import safe_cuda_cache_empty
from utils.image import cv_save_image, img2tensor, tensor2img, read_cv
from utils.tile import auto_split


class Upscaler:
    def __init__(self, model_path, input_folder, output_folder):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(model_path, map_location='cpu')

        model = load_model(state_dict)
        model.eval()
        model = model.to(device)

        self.model = model
        self.device = device
        self.input_folder = input_folder
        self.output_folder = output_folder

        print(f"Model Architecture: {model.name}")

    def __upscale(self, img: np.ndarray) -> np.ndarray:
        tensor = img2tensor(img)

        rgba_arch = True if self.model.name == "DITN" else False
        if rgba_arch:
            tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            tensor = self.model(tensor.to(self.device))

        return tensor2img(tensor)

    def run(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        for filename in os.listdir(self.input_folder):
            input_image_path = os.path.join(self.input_folder, filename)
            try:
                img = read_cv(input_image_path)
                if img is None:
                    raise RuntimeError(f"Unsupported image type: {filename}")

                result = auto_split(img, self.__upscale, (128, 128))

                output_image_path = os.path.join(self.output_folder, filename)
                cv_save_image(output_image_path, result, [])

            except RuntimeError as e:
                print(f"[FAILED] {filename} : {e}")
            else:
                print(f"[DONE] {filename}")

        safe_cuda_cache_empty()
