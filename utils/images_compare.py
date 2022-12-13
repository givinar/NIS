import torch
from torchmetrics import MeanAbsolutePercentageError
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
from typing import Tuple

"""
Module compares two sequences of images from two different folders with ground truth image
using specified metric and shows its in the plot.
Example: -im 'image_folder_path' -gt 'ground_truth_image_path' -s 50
"""


@dataclass
class Metrics:
    gt_image: str

    def __post_init__(self):
        self.gt_tsr = self.load_exr(self.gt_image)
        self.gt_img = Image.fromarray(np.uint8(self.gt_tsr * 255.))

    def ssim(self, preds: np.ndarray) -> float:
        preds_img = Image.fromarray(np.uint8(preds * 255.))
        return compare_ssim(preds_img, self.gt_img)

    def mape(self, preds: np.ndarray) -> float:
        return MeanAbsolutePercentageError()(torch.from_numpy(preds).to(torch.float),
                                             torch.from_numpy(self.gt_tsr).to(torch.float))

    @staticmethod
    def load_exr(path, max_channels=3):
        path = str(path)
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


@dataclass
class ImageCompare:
    img_folder: str
    gt_image: str
    metric_name: str
    step: int

    def __post_init__(self):
        self.metrics = Metrics(self.gt_image)
        self.set_metric(self.metric_name)

    @staticmethod
    def parse_args(arg=sys.argv[1:]):
        arg_parser = argparse.ArgumentParser(
            description='Comparing images by specific metric')

        arg_parser.add_argument('-im', '--image_folder', type=str, required=True, help='Image folder path'),
        arg_parser.add_argument('-gt', '--image_gt', type=str, required=True, help='Ground truth image path')
        arg_parser.add_argument('-m', '--metric', type=str, required=True, help='Metric')
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

    def calc_metric(self, path: str, length: int, method: Callable) -> Tuple[np.ndarray, np.ndarray]:
        metric = np.zeros(length, float)
        indices = np.zeros(length, int)
        for i, name in enumerate(self.list_files(path)):
            metric[i] = method(self.metrics.load_exr(name))
            indices[i] = i * self.step

        return indices, metric

    def show_plot(self):
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        plt.figure(self.metric_name)
        try:
            for i, folder in enumerate(os.listdir(self.img_folder)):
                full_path = os.path.join(self.img_folder, folder)
                length = len(os.listdir(full_path))
                indices, ssims = self.calc_metric(full_path, length, self.metric)
                plt.plot(indices, ssims, colors[i], label=os.path.basename(full_path))
        except IndexError:
            print('Color index error, extend color list...')
            raise SystemExit

        plt.xlabel('Iterations')
        plt.ylabel(self.metric_name)
        #plt.ylim(0., 1.)
        plt.grid()
        plt.legend()
        plt.show()

    def set_metric(self, opt):
        if opt == 'SSIM':
            self.metric = self.metrics.ssim
        elif opt == 'MAPE':
            self.metric = self.metrics.mape
        else:
            print('Unknown metric type -', opt)
            raise SystemExit


if __name__ == '__main__':
    options = ImageCompare.parse_args()

    ic = ImageCompare(options.image_folder,
                      options.image_gt,
                      options.metric,
                      options.step)

    ic.show_plot()
