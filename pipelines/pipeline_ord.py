import wandb
import os.path as osp
import torch
import numpy as np

from pytorch_lightning.loggers import WandbLogger

from data_prep import prepare_data, add_replay_data
from gp.lightning.data_template import DataWithMeta, DataModule
from gp.lightning.module_template import ExpConfig
from gp.lightning.training import lightning_fit
from pipelines.setups.metric_prepare import build_eval_kit
from pipelines.setups.model_prepare import (
    prepare_model_rl,
    prepare_agent,
)
from pipelines.setups.lightning_prepare import (
    prepare_lightning_rl,
    prepare_lightning_rand_NM,
    prepare_lightning_rl_pred,
)

from pipelines.setups.function_setup import safe_load_create_env, safe_load_create_mover
from utils import log_mean_var


def main(params):
    data = prepare_data(
        params, params.data_path, params.train_data_set, *params.data_arg
    )
    for dt in params.data_trans:
        dt(data, params)

    pred_datasets = {
        "train": DataWithMeta(data["train"], params.batch_size, "train", sample_size=params.train_sample_size),
        "val": [
            DataWithMeta(data["valid"], params.batch_size, "valid", meta_data={"eval_func": params.eval_func},
                         metric=params.metric, classes=data["num_class"], sample_size=params.eval_sample_size)],
        "test": [
            DataWithMeta(data["test"], params.batch_size, "test", meta_data={"eval_func": params.eval_func},
                         metric=params.metric, classes=data["num_class"], sample_size=params.eval_sample_size)]}

    pred_data_module = DataModule(pred_datasets, params.num_workers)

    rl_datasets = {
        "train": [DataWithMeta(data["train"], params.batch_size, "exp_train", sample_size=params.train_sample_size),
                  DataWithMeta(data["replay_train"], params.batch_size, "replay_train",
                               sample_size=params.train_sample_size)],
        "val": [
            DataWithMeta(data["valid"], params.batch_size, "valid", meta_data={"eval_func": params.eval_func},
                         metric=params.metric, classes=data["num_class"], sample_size=params.eval_sample_size)],
        "test": [
            DataWithMeta(data["test"], params.batch_size, "test", meta_data={"eval_func": params.eval_func},
                         metric=params.metric, classes=data["num_class"], sample_size=params.eval_sample_size)]}

    std = data["std"] if "std" in data else None
    rl_data_module = DataModule(rl_datasets, params.num_workers)

    torch_models = safe_load_create_env(params, data)

    rl_models, rl_target_models = safe_load_create_mover(params, data)

    wandb_logger = WandbLogger(
        project=params.log_project,
        name=params.exp_name,
        save_dir=params.exp_dir,
        offline=params.offline_log,
    )

    agent = prepare_agent(
        params, data, rl_models, torch_models, params.agent_eval
    )

    rnm_name = "rnm"
    rl_name = "rl"
    rl_pred_name = "rl_pred"

    # Determines whether to train the prediction GNN using random node markings first

    if params.train_pred_gnn and not hasattr(params, "env_load"):
        pred_eval_kit = build_eval_kit(pred_datasets, params, rnm_name, eval_train=True, std=std)
        pred_optim = torch.optim.Adam(
            torch_models.parameters(), lr=params.lr, weight_decay=params.l2
        )
        exp_config = ExpConfig(rnm_name, pred_optim)
        exp_config.val_state_name = [dm.state_name for dm in pred_datasets["val"]]
        exp_config.test_state_name = [dm.state_name for dm in pred_datasets["test"]]

        lightning_model = prepare_lightning_rand_NM(
            params, torch_models, exp_config, pred_eval_kit, rnm_name
        )

        val_res, test_res = lightning_fit(
            wandb_logger,
            lightning_model,
            pred_data_module,
            pred_eval_kit,
            params.num_epochs,
            cktp_prefix=rnm_name + "-",
            load_best=params.load_best,
        )
        log_mean_var(wandb_logger, pred_eval_kit.test_metric, *test_res)
        log_mean_var(wandb_logger, pred_eval_kit.val_metric, *val_res)

    train_rl = not params.only_train_pred     # determines whether the RL agent should be trained in the episode
    train_rl_pred = params.last_train_pred or params.episode > 1    # Determines whether the pred gnn should be trained

    for i in range(params.episode):
        if train_rl:
            rl_episode_name = rl_name + "/" + str(i)
            rl_eval_kit = build_eval_kit(rl_datasets, params, rl_episode_name, std=std)
            rl_optim = torch.optim.Adam(rl_models.parameters(), lr=params.lr, weight_decay=params.rl_l2)

            rl_target_models.load_state_dict(rl_models.state_dict())

            exp_config = ExpConfig(rl_episode_name, rl_optim)
            exp_config.val_state_name = [dm.state_name for dm in rl_datasets["val"]]
            exp_config.test_state_name = [dm.state_name for dm in rl_datasets["test"]]
            rl_lightning_model = prepare_lightning_rl(params, torch_models, rl_models, rl_target_models, agent,
                                                      exp_config,
                                                      rl_eval_kit, rl_episode_name)
            rl_lightning_model.warm_up(params.batch_size * params.replay_size * 5, rl_datasets["train"][0])

            val_res, test_res = lightning_fit(
                wandb_logger,
                rl_lightning_model,
                rl_data_module,
                rl_eval_kit,
                params.num_rl_epochs,
                cktp_prefix=rl_name + "-",
            )
            log_mean_var(wandb_logger, rl_eval_kit.test_metric, *test_res)
            log_mean_var(wandb_logger, rl_eval_kit.val_metric, *val_res)

        if train_rl_pred:
            rl_pred_episode_name = rl_pred_name + "/" + str(i)
            rl_pred_eval_kit = build_eval_kit(pred_datasets, params, rl_pred_episode_name, std=std)
            pred_optim = torch.optim.Adam(torch_models.parameters(), lr=params.lr, weight_decay=params.l2)

            exp_config = ExpConfig(rl_pred_episode_name, pred_optim)
            exp_config.val_state_name = [dm.state_name for dm in rl_datasets["val"]]
            exp_config.test_state_name = [dm.state_name for dm in rl_datasets["test"]]

            rl_pred_lightning_model = prepare_lightning_rl_pred(params, torch_models, rl_models, agent, exp_config,
                                                                rl_pred_eval_kit,
                                                                rl_pred_episode_name)

            val_res, test_res = lightning_fit(
                wandb_logger,
                rl_pred_lightning_model,
                pred_data_module,
                rl_pred_eval_kit,
                params.num_epochs,
                cktp_prefix=rl_pred_name + "-",
            )
            log_mean_var(wandb_logger, rl_pred_eval_kit.test_metric, *test_res)
            log_mean_var(wandb_logger, rl_pred_eval_kit.val_metric, *val_res)
        train_rl = True
        train_rl_pred = params.last_train_pred or i < (params.episode - 2)

    wandb.finish()
