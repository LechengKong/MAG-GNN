import argparse
import pipelines.pipeline_rl_simul
import pipelines.pipeline_ord

import torch

from types import SimpleNamespace
from pipelines.setups.function_setup import make_functions_from_params
from utils import setup_exp
from gp.utils.utils import set_random_seed, combine_dict, merge_mod
from utils import load_yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rl")
    parser.add_argument(
        "--data_config_path",
        type=str,
        default="./configs/dataset_configs/srg.yaml",
    )
    parser.add_argument(
        "--exp_config_path",
        type=str,
        default="./configs/exp_configs/simul.yaml",
    )
    parser.add_argument("--override", type=str)
    parser.add_argument(
        "opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line",
    )
    params = parser.parse_args()
    configs = []
    configs.append(load_yaml("./configs/default_config.yaml"))
    configs.append(load_yaml("./configs/rl_config.yaml"))
    configs.append(load_yaml(params.data_config_path))
    configs.append(load_yaml(params.exp_config_path))
    if params.override is not None:
        override_config = load_yaml(params.override)
        configs.append(override_config)

    mod_params = combine_dict(*configs)
    mod_params = merge_mod(mod_params, params.opts)
    setup_exp(mod_params)

    params = SimpleNamespace(**mod_params)
    set_random_seed(params.seed)

    make_functions_from_params(params)
    torch.set_float32_matmul_precision("high")

    # Using different pipelines for SIMUL training and other form of training

    if params.training_mode == "simul":
        params.log_project = "simul_" + params.train_data_set
        pipelines.pipeline_rl_simul.main(params)
    else:
        params.log_project = params.training_mode + "_" + params.train_data_set
        pipelines.pipeline_ord.main(params)
