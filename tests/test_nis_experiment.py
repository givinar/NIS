import shutil
from experiment_config import ExperimentConfig
from nis import NeuralImportanceSampling
from pathlib import Path

TMP_TEST_PLOTS_DIR = Path('logs', 'tmp_test')
OUTPUT_TEST_PLOTS_DIR = Path('test_outputs')
if OUTPUT_TEST_PLOTS_DIR.exists():
    shutil.rmtree(OUTPUT_TEST_PLOTS_DIR)
OUTPUT_TEST_PLOTS_DIR.mkdir(exist_ok=True)
BASE_CONFIG = ExperimentConfig(ndims=2,
                               epochs=100,
                               batch_size=1000,
                               lr=0.001,
                               hidden_dim=256,
                               n_hidden_layers=6,
                               piecewise_bins=32,
                               loss_func='KL',
                               save_plots=True,
                               save_plt_interval=100,
                               num_context_features=0,
                               experiment_dir_name=TMP_TEST_PLOTS_DIR.name
                               )
TEST_COUPLING_LAYERS = ['piecewiseQuadratic', 'piecewiseLinear']
TEST_BLOB_BINS = [0, 10]
TEST_NETS = ['unet', 'mlp']
TEST_FUNCTIONS = ['ImageFunc', 'Gaussian']


def test_experiment_configs():
    for num_blob_bins in TEST_BLOB_BINS:
        for layer_type in TEST_COUPLING_LAYERS:
            for function in TEST_FUNCTIONS:
                for net in TEST_NETS:
                    BASE_CONFIG.blob = num_blob_bins
                    BASE_CONFIG.coupling_name = layer_type
                    BASE_CONFIG.funcname = function
                    BASE_CONFIG.network_type = net

                    nis = NeuralImportanceSampling(BASE_CONFIG)
                    nis.initialize()
                    nis.run_experiment()

                    test_output_img_path = next((TMP_TEST_PLOTS_DIR / 'plots').iterdir())
                    test_output_img_path.rename(OUTPUT_TEST_PLOTS_DIR /
                                f'{function}_{net}_{layer_type}__{num_blob_bins}blobs.png')
