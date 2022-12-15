import argparse
import json
import logging
import sys
import subprocess
import time
from multiprocessing import Process
import os

from integration_experiment import ExperimentConfig, server_processing
from sklearn.model_selection import ParameterGrid

from utils import pyhocon_wrapper
from utils.images_compare import Metrics

GRID_PARAMS = {
    "funcname": ["Gaussian"],
    "coupling_name": ["piecewiseQuadratic"],
    "hidden_dim": [10],
    "n_hidden_layers": [3],
    "blob": [None],
    "piecewise_bins": [10],
    "lr": [0.1, 0.01, 0.001, 0.0001],
    "loss_func": ["MSE"],
    "gradient_accumulation": [True],
    "network_type": ['unet']  # ['mlp', 'unet'],
}
HYBRID_ARGS = '-w 640 -h 480'
GRID_SEARCH_OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'grid_search'
)
SAVE_INTERVAL = 50
NUM_FRAMES = 1


def run_experiment(base_config: ExperimentConfig, config_params, standalone_path: str, metric_calculation):
    base_config.funcname = config_params['funcname']
    base_config.hidden_dim = config_params['hidden_dim']
    base_config.n_hidden_layers = config_params['n_hidden_layers']
    base_config.blob = config_params['blob']
    base_config.piecewise_bins = config_params['piecewise_bins']
    base_config.lr = config_params['lr']
    base_config.loss_func = config_params['loss_func']
    base_config.gradient_accumulation = config_params['gradient_accumulation']
    # base_config.network_type = config_params['network_type']  # TODO: uncomment after merge #14

    server_process = Process(target=server_processing, args=(base_config,))
    server_process.start()

    dir_name = 'test_run'

    standalone_dir_path = os.path.abspath(os.path.dirname(standalone_path))
    experiment_dir_path = os.path.join(standalone_dir_path, GRID_SEARCH_OUT_DIR, dir_name)
    os.makedirs(experiment_dir_path, exist_ok=True)
    image_name = os.path.join(GRID_SEARCH_OUT_DIR, dir_name, "grid_search_{frame_number}")
    hybrid_process = subprocess.Popen(
        f'{standalone_path} --nsi={SAVE_INTERVAL} --ifn={image_name} {HYBRID_ARGS}',
        cwd=standalone_dir_path)

    while len(os.listdir(experiment_dir_path)) < NUM_FRAMES:
        time.sleep(5)

    hybrid_process.kill()
    server_process.kill()

    #  calculate last image metric
    imgs = os.listdir(experiment_dir_path)
    imgs_paths = [os.path.join(experiment_dir_path, basename) for basename in imgs]
    last_image_path = max(imgs_paths, key=os.path.getctime)
    last_image = metric_calculation.load_exr(last_image_path)

    metric = metric_calculation.ssim(last_image)
    config_params['metric'] = metric
    with open(os.path.join(experiment_dir_path, "params.json"), "w") as outfile:
        json.dump(config_params, outfile, indent=4)

    out_dir_name = f'{metric:.4f}_{time.strftime("%Y%m%d_%H%M%S")}'
    os.rename(
        os.path.join(standalone_dir_path, GRID_SEARCH_OUT_DIR, dir_name),
        os.path.join(standalone_dir_path, GRID_SEARCH_OUT_DIR, out_dir_name),
    )


def parse_args(arg=sys.argv[1:]):
    arg_parser = argparse.ArgumentParser(
        description='Application for grid search')

    arg_parser.add_argument(
        '-c', '--config', required=True,
        help='Base configuration file path')

    arg_parser.add_argument(
        '-s', '--standalone', default=os.path.join('..', 'BaikalNext_2', 'bin', 'Standalone.exe'),
        help='Standalone.bat file path')
    arg_parser.add_argument('-gt', '--image_gt', type=str, required=True, help='Ground truth image path for metric')

    return arg_parser.parse_args(arg)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)

    options = parse_args()
    base_config = ExperimentConfig.init_from_pyhocon(
        pyhocon_wrapper.parse_file(options.config)
    )
    grid = ParameterGrid(GRID_PARAMS)
    metric_calculation = Metrics(gt_image=options.image_gt)
    for config_params in grid:
        run_experiment(base_config, config_params, options.standalone, metric_calculation)
