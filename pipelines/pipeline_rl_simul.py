import wandb

import torch

from pytorch_lightning.loggers import WandbLogger

from data_prep import prepare_data
from pipelines.setups.model_prepare import (
    prepare_model_rl,
    prepare_agent,
)
from pipelines.setups.lightning_prepare import prepare_lightning_rl
from utils import log_step

from gp.lightning.data_template import DataModule, DataWithMeta
from gp.lightning.module_template import ExpConfig
from gp.lightning.training import lightning_fit

from pipelines.setups.function_setup import safe_load_create_env
from pipelines.setups.metric_prepare import build_eval_kit


def main(params):
    data = prepare_data(
        params, params.data_path, params.train_data_set
    )
    for dt in params.data_trans:
        dt(data, params)

    datasets = {
        "train": [DataWithMeta(data["train"], params.batch_size, "exp_train", sample_size=params.train_sample_size),
                  DataWithMeta(data["replay_train"], params.batch_size, "replay_train",
                               sample_size=params.train_sample_size),
                  DataWithMeta(data["train"], params.batch_size, "pred_train",
                               meta_data={"eval_func": params.eval_func}, metric=params.metric,
                               classes=data["num_class"], sample_size=params.train_sample_size)],
        "val": [
            DataWithMeta(data["valid"], params.batch_size, "valid", meta_data={"eval_func": params.eval_func},
                         metric=params.metric, classes=data["num_class"], sample_size=params.eval_sample_size)],
        "test": [
            DataWithMeta(data["test"], params.batch_size, "test", meta_data={"eval_func": params.eval_func},
                         metric=params.metric, classes=data["num_class"], sample_size=params.eval_sample_size)]}
    data_module = DataModule(datasets, params.num_workers)
    std = data["std"] if "std" in data else None

    torch_models = safe_load_create_env(params, data)
    rl_models = prepare_model_rl(params, data)
    rl_target_models = prepare_model_rl(params, data)

    eval_kit = build_eval_kit(datasets, params, "", eval_train=True, std=std)

    agent = prepare_agent(
        params, data, rl_models, torch_models, params.agent_eval
    )

    rl_target_models.load_state_dict(rl_models.state_dict())

    pred_optim = torch.optim.Adam(
        torch_models.parameters(), lr=params.lr, weight_decay=params.l2
    )

    rl_optim = torch.optim.Adam(rl_models.parameters(), lr=params.lr, weight_decay=params.rl_l2)

    exp_config = ExpConfig("rl", [rl_optim, pred_optim])

    rl_lightning_model = prepare_lightning_rl(params, torch_models, rl_models, rl_target_models, agent, exp_config,
                                              eval_kit, "rl")
    rl_lightning_model.warm_up(params.batch_size * params.replay_size * 5, datasets["train"][0])

    wandb_logger = WandbLogger(
        project=params.log_project,
        name=params.exp_name,
        save_dir=params.exp_dir,
        offline=params.offline_log,
    )

    _, test_res = lightning_fit(
        wandb_logger,
        rl_lightning_model,
        data_module,
        eval_kit,
        params.num_rl_epochs,
        cktp_prefix="rl-",
    )
    log_step(wandb_logger, params.metric, "rl", *test_res)

    wandb.finish()
