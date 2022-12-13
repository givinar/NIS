from dataclasses import dataclass
from collections.abc import Callable
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import Imath
import OpenEXR
import argparse
from PIL import Image
from SSIM_PIL import compare_ssim

"""
Module compares two sequences of images from two different folders with ground truth image
using specified metric and shows its in the plot.
Example: -im 'image_folder_path' -gt 'ground_truth_image_path' -s 50
"""

@dataclass
class ImageCompare:
    img_folder: str
    gt_image: str
    step: int

    def __post_init__(self):
        gt_tsr = self.load_exr(self.gt_image)
        self.gt_pil_img = Image.fromarray(np.uint8(gt_tsr * 255.))

    @staticmethod
    def parse_args(arg=sys.argv[1:]):
        arg_parser = argparse.ArgumentParser(
            description='Comparing images by specific metric')

        arg_parser.add_argument('-im', '--image_folder', type=str, required=True, help='Image folder path'),
        arg_parser.add_argument('-gt', '--image_gt', type=str, required=True, help='Ground truth image path')
        arg_parser.add_argument('-s', '--step', type=int, required=True, help='Iteration step')

        return arg_parser.parse_args(arg)

    @staticmethod
    def list_files(path, valid_exts='exr'):
        for (rootDir, dirNames, filenames) in os.walk(path):
            for filename in filenames:
                ext = filename[filename.rfind("."):].lower()
                if ext.endswith(valid_exts):
                    image_path = os.path.join(rootDir, filename).replace(" ", "\\ ")
                    yield image_path

    @staticmethod
    def load_exr(path, max_channels=3):
        path = str(path)  # convert from Path if needed
        if not OpenEXR.isOpenExrFile(path):
            raise ValueError(f'Image {path} is not a valid OpenEXR file')
        src = OpenEXR.InputFile(path)
        header = src.header()
        num_channels = min(len(src.header()['channels']),
                           max_channels)
        dw = header['dataWindow']
        min_y = dw.min.y
        max_y = dw.max.y
        size = (dw.max.x - dw.min.x + 1, max_y - min_y + 1)

        if header['channels']['R'] == Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)):
            np_dtype = np.float16
        elif header['channels']['R'] == Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)):
            np_dtype = np.float32
        else:
            assert False, 'Unknown pixel type in EXR header'

        np_data = np.empty((num_channels, size[1] * size[0]), dtype=np_dtype)
        buffer_list = src.channels('RGBA'[:num_channels], scanLine1=min_y, scanLine2=max_y)

        for c, buffer in enumerate(buffer_list):
            np_data[c] = np.frombuffer(buffer, dtype=np_dtype)

        tensor = np_data.reshape((-1, size[1], size[0]))
        return tensor.transpose(1, 2, 0)

    def get_ssim(self, path: str, length: int, method: Callable) -> np.ndarray:
        ssims = np.zeros(length, float)
        indices = np.zeros(length, int)
        for i, name in enumerate(self.list_files(path)):
            tsr = self.load_exr(name)
            pil_img = Image.fromarray(np.uint8(tsr * 255))
            ssims[i] = method(pil_img, self.gt_pil_img)
            indices[i] = i * self.step

        return indices, ssims

    def show_plot(self):
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        plt.figure('SSIM')
        try:
            for i, folder in enumerate(os.listdir(self.img_folder)):
                full_path = os.path.join(self.img_folder, folder)
                length = len(os.listdir(full_path))
                indices, ssims = self.get_ssim(full_path, length, compare_ssim)
                plt.plot(indices, ssims, colors[i], label=os.path.basename(full_path))
        except IndexError:
            print('Color index error, extend color list...')
            raise SystemExit

        plt.xlabel('Iterations')
        plt.ylabel('SSIM')
        #plt.ylim(0., 1.)
        plt.grid()
        plt.legend()
        plt.show()


if __name__ == '__main__':
    options = ImageCompare.parse_args()

    ic = ImageCompare(options.image_folder,
                      options.image_gt,
                      options.step)
    ic.show_plot()
