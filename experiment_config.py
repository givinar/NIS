from dataclasses import dataclass
from datetime import datetime
from typing import Union
from utils import pyhocon_wrapper


@dataclass
class ExperimentConfig:
    """
    experiment_dir_name: dir in logs folder
    ndims: Integration dimension
    funcname: Name of the function in functions.py to use for integration
    transform_name: name of the Coupling Layers using in NIS [piecewiseLinear, piecewiseQuadratic, piecewiseCubic]
    num_context_features: : number of context features in transform net
    hidden_dim: Number of neurons per layer in the coupling layers
    n_hidden_layers: Number of hidden layers in coupling layers
    blob: Number of bins for blob-encoding (default = None)
    piecewise_bins: Number of bins for piecewise polynomial coupling (default = 10)
    lr: Learning rate
    epochs: Number of epochs
    loss_func: Name of the loss function in divergences (default = MSE)
    batch_size: Batch size
    save_plots: save plots if ndims >= 2
    plot_dimension: add 2d or 3d plot
    save_plt_interval: Frequency for plot saving (default : 10)
    wandb_project: Name of wandb project in neural_importance_sampling team
    use_tensorboard: Use tensorboard logging
    """

    experiment_dir_name: str = f"test_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    ndims: int = 3
    funcname: str = "Gaussian"
    transform_name: str = "piecewiseQuadratic"
    num_context_features: int = 0
    hidden_dim: int = 10
    n_hidden_layers: int = 3
    network_type: str = "MLP"
    blob: Union[int, None] = None
    piecewise_bins: int = 10
    lr: float = 0.01
    epochs: int = 100
    loss_func: str = "MSE"
    batch_size: int = 2000
    gradient_accumulation: bool = True
    hybrid_sampling: bool = False
    num_training_steps: int = 16
    num_samples_per_training_step: int = 10_000
    max_train_buffer_size: int = 2_000_000
    features_mode: str = "all_features"  # 'no_features' 'xyz' 'all_features'
    one_bounce_mode: bool = True

    num_mixtures: int = 0

    save_plots: blob = True
    plot_dimension: int = 2
    save_plt_interval: int = 10
    wandb_project: Union[str, None] = None
    use_tensorboard: bool = False
    host: str = "127.0.0.1"
    port: int = 65432

    @classmethod
    def init_from_pyhocon(cls, pyhocon_config: pyhocon_wrapper.ConfigTree):
        return ExperimentConfig(
            epochs=pyhocon_config.get_int("train.epochs"),
            batch_size=pyhocon_config.get_int("train.batch_size"),
            gradient_accumulation=pyhocon_config.get_bool(
                "train.gradient_accumulation", True
            ),
            lr=pyhocon_config.get_float("train.learning_rate"),
            hidden_dim=pyhocon_config.get_int("train.num_hidden_dims"),
            ndims=pyhocon_config.get_int("train.num_coupling_layers"),
            n_hidden_layers=pyhocon_config.get_int("train.num_hidden_layers"),
            network_type=pyhocon_config.get_string("train.network_type", "MLP"),
            blob=pyhocon_config.get_int("train.num_blob_bins", 0),
            piecewise_bins=pyhocon_config.get_int("train.num_piecewise_bins", 10),
            loss_func=pyhocon_config.get_string("train.loss", "MSE"),
            save_plt_interval=pyhocon_config.get_int("logging.save_plt_interval", 5),
            experiment_dir_name=pyhocon_config.get_string(
                "logging.plot_dir_name", cls.experiment_dir_name
            ),
            hybrid_sampling=pyhocon_config.get_bool("train.hybrid_sampling", False),
            num_training_steps=pyhocon_config.get_int("train.num_training_steps", 16),
            num_samples_per_training_step=pyhocon_config.get_int(
                "train.num_samples_per_training_step", 10_000
            ),
            max_train_buffer_size=pyhocon_config.get_int(
                "train.max_train_buffer_size", 2_000_000
            ),
            features_mode=pyhocon_config.get_string(
                "train.features_mode", "all_features"
            ),
            one_bounce_mode=pyhocon_config.get_bool("train.one_bounce_mode", True),

            num_mixtures=pyhocon_config.get_int("train.num_mixtures", 0),

            funcname=pyhocon_config.get_string("train.function"),
            transform_name=pyhocon_config.get_string("train.transform_name"),
            num_context_features=pyhocon_config.get_int("train.num_context_features"),
            wandb_project=pyhocon_config.get_string(
                "logging.tensorboard.wandb_project", None
            ),
            use_tensorboard=pyhocon_config.get_bool(
                "logging.tensorboard.use_tensorboard", False
            ),
            save_plots=pyhocon_config.get_bool("logging.save_plots", False),
            plot_dimension=pyhocon_config.get_int("logging.plot_dimension", 2),
            host=pyhocon_config.get_string("server.host", "127.0.0.1"),
            port=pyhocon_config.get_int("server.port", 65432),
        )
