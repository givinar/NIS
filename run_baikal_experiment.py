import logging
import shutil
import subprocess
from multiprocessing import Process
import os

from server_nis import server_processing
from experiment_config import ExperimentConfig

from utils import pyhocon_wrapper

EXPERIMENT_NAME = "test_experiment"
EXPERIMENT_OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "compare_imgs", "10_presentation_10.01"
)
CONFIG_PATH = os.path.join("config", "config.conf")


NUM_ITERATIONS = 1000
SAVE_INTERVAL = 1
NUM_BOUNCE = 3
RESOLUTION = [10, 10]


STANDALONE_PATH = os.path.join("..", "BaikalNext_mis", "bin", "Standalone.exe")
HYBRID_ARGS = (
    f"-w {RESOLUTION[0]} -h {RESOLUTION[1]} --rt_max_recursion={NUM_BOUNCE} "
    f"--nsi={SAVE_INTERVAL} --no-ui --ns={NUM_ITERATIONS} "
    '--use-cameras="..\..\VeachAjar_GLTF\camera"  -p "..\..\VeachAjar_GLTF" -f "VeachAjar.gltf"'
)


def run_experiment(config: ExperimentConfig, standalone_path: str):
    server_process = Process(target=server_processing, args=(config,))
    server_process.start()

    dir_name = EXPERIMENT_NAME

    standalone_dir_path = os.path.abspath(os.path.dirname(standalone_path))
    experiment_dir_path = os.path.join(
        standalone_dir_path, EXPERIMENT_OUT_DIR, dir_name
    )
    os.makedirs(experiment_dir_path, exist_ok=True)
    image_name = os.path.join(EXPERIMENT_OUT_DIR, dir_name, "experiment_{frame_number}")
    hybrid_process = subprocess.Popen(
        f"{standalone_path} --ifn={image_name} {HYBRID_ARGS}", cwd=standalone_dir_path
    )

    hybrid_process.wait()

    #  calculate last image metric
    imgs = os.listdir(experiment_dir_path)
    imgs_paths = [os.path.join(experiment_dir_path, basename) for basename in imgs]
    imgs = sorted(imgs_paths, key=os.path.getctime)

    pos_path = os.path.join(EXPERIMENT_OUT_DIR, f"{dir_name}_pos")
    os.makedirs(pos_path, exist_ok=True)
    for img in imgs[NUM_ITERATIONS:]:
        to_img = os.path.join(pos_path, os.path.basename(img))
        shutil.move(img, to_img)


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.WARNING)

    base_config = ExperimentConfig.init_from_pyhocon(
        pyhocon_wrapper.parse_file(CONFIG_PATH)
    )
    run_experiment(base_config, STANDALONE_PATH)
