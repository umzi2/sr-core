import os
import numpy as np
import torch
from archs import load_model
from utils.cuda import safe_cuda_cache_empty
from utils.image import cv_save_image, img2tensor, tensor2img, read_cv
from utils.tile import auto_split
from tqdm import tqdm
from utils.unpickler import RestrictedUnpickle
from moviepy.editor import VideoFileClip


class Upscaler:
    def __init__(self, model_path, input_folder, output_folder, tile_size=256, form="png"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(
            model_path, map_location="cpu", pickle_module=RestrictedUnpickle
        )

        model = load_model(state_dict)
        model.eval()
        model = model.to(device)

        self.model = model
        self.device = device
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.tile_max_size = tile_size
        self.format_image = form
        if self.model.input_channels == 1:
            self.channels = "grayscale"
        else:
            self.channels = "color"

    def __upscale(self, img: np.ndarray) -> np.ndarray:
        tensor = img2tensor(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            tensor = self.model(tensor)

        return tensor2img(tensor)

    def run(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        list_files = [
            file
            for file in os.listdir(self.input_folder)
            if os.path.isfile(os.path.join(self.input_folder, file))
        ]
        for filename in tqdm(list_files, desc=self.model.name, leave=True):
            input_image_path = os.path.join(self.input_folder, filename)
            try:
                img = read_cv(input_image_path, self.channels)
                if img is None:
                    raise RuntimeError(f"Unsupported image type: {filename}")

                result = auto_split(img, self.tile_max_size, self.__upscale)
                output_image_path = os.path.join(self.output_folder, "".join(filename.split(".")[:-1]))
                output_image_path_format = f"{ output_image_path }.{self.format_image}"
                cv_save_image(output_image_path_format, result, [])

            except RuntimeError as e:
                print(f"[FAILED] {filename} : {e}")

        safe_cuda_cache_empty()


class UpscalerVideo:
    def __init__(self, model_path, input_folder, output_folder, tile_size=256, form_video="mp4", codec_video="libx264",
                 codec_audio="aac"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(
            model_path, map_location="cpu", pickle_module=RestrictedUnpickle
        )

        model = load_model(state_dict)
        model.eval()
        model = model.to(device)

        self.model = model
        self.device = device
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.tile_max_size = tile_size
        self.format_video = form_video
        self.codec_video = codec_video
        self.codec_audio = codec_audio
        if self.model.input_channels == 1:
            self.channels = "grayscale"
        else:
            self.channels = "color"

    def __upscale(self, img: np.ndarray) -> np.ndarray:
        tensor = img2tensor(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            tensor = self.model(tensor)

        return tensor2img(tensor)

    def process_frame(self, frame):

        frame_np = np.array(frame) / 255
        return auto_split(frame_np, self.tile_max_size, self.__upscale)

    def run(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        list_files = [
            file
            for file in os.listdir(self.input_folder)
            if os.path.isfile(os.path.join(self.input_folder, file))
        ]
        for filename in list_files:
            input_video_path = os.path.join(self.input_folder, filename)
            try:
                video_clip = VideoFileClip(input_video_path)
                processed_clip = video_clip.fl_image(self.process_frame)
                output_video_path = os.path.join(self.output_folder, "".join(filename.split(".")[:-1]))
                output_video_path_format = f"{output_video_path}.{self.format_video}"

                processed_clip.write_videofile(output_video_path_format, codec=self.codec_video,
                                               audio_codec=self.codec_audio)
            except RuntimeError as e:
                return print(f"[FAILED] {e}")
        safe_cuda_cache_empty()

