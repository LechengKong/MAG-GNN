import torch
import os.path as osp
import numpy as np
import wandb
from gp.utils.utils import SmartTimer
from dgl.dataloading.dataloader import GraphCollator
from gp.lightning.module_template import BaseTemplate
from kernel.lightning_template import QLTemplate
from utils import random_label, nm_init
from tqdm import tqdm
from gp.lightning.data_template import create_dataloader


class GraphRandLabelPredTemplate(BaseTemplate):
    def __init__(self, exp_config, model, eval_kit, num_nm, num_piece, name=""):
        super().__init__(exp_config, model, eval_kit, name)
        self.num_nm = num_nm
        self.num_piece = num_piece

    def forward(self, batch):
        g = batch.g
        labels = nm_init(g, self.num_nm, self.num_piece)
        pred = self.model["task_predictor"](g, labels)
        pred = self.model["classifier"](pred)
        return pred


class GraphFixedLabelPredTemplate(BaseTemplate):
    def forward(self, batch):
        g = batch.g
        pred = self.model["task_predictor"](g, batch.rlabel)
        pred = self.model["classifier"](pred)
        return pred


class QLMoverTrainOrd(QLTemplate):
    def __init__(self, exp_config, environment, mover, target_mover, agent, k, eval_kit, name="rl", rl_config=None,
            **kwargs, ):
        super().__init__(exp_config, environment, mover, agent, k, eval_kit, name, )
        self.steps = 0
        self.eps_reset_count = 0
        self.reset_flag = False
        self.rl_config = rl_config
        self.target_mover = target_mover
        self.update_target()

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        exp_batch, replay_batch = batch
        self.experience_state()
        eps = self.get_epsilon()
        loss = self.exp_replay(exp_batch, replay_batch, eps)
        self.log(osp.join(self.name, "train/loss"), loss if loss is not None else 0, prog_bar=True,
            batch_size=replay_batch.g.batch_size, )
        self.log(osp.join(self.name, "eps"), eps, batch_size=replay_batch.g.batch_size, )
        self.steps += 1
        self.update_target()
        return loss

    def exp_replay(self, exp_batch, replay_batch, eps):
        reward = self.agent.experience_fn(self.mover, self.model, eps, exp_batch)
        mean_reward = torch.mean(reward)
        self.log(osp.join(self.name, "reward"), mean_reward, prog_bar=True, batch_size=exp_batch.g.batch_size, )
        self.replay_state()
        loss = self.replay(replay_batch)
        return loss

    def warm_up(self, minimum_buffer_size, data, prog_bar=True, num_workers=0):
        self.experience_state()
        warm_up_loader = self.warmup_dataloader(data, minimum_buffer_size, num_workers)
        pbar = tqdm(warm_up_loader) if prog_bar else warm_up_loader
        eps = self.get_epsilon()
        for batch in pbar:
            batch = self.transfer_batch_to_device(batch, self.device, dataloader_idx=0)
            self.agent.experience_fn(self.mover, self.model, eps, batch)

    def replay(self, memory):

        q_val, max_q = self.agent.replay(self.mover, self.target_mover, memory)
        target = max_q * self.rl_config["gamma"] + memory.reward
        res = torch.mean((q_val - target) ** 2)
        return res

    def get_epsilon(self):
        if self.steps < self.rl_config["eps_first_frame"]:
            return self.rl_config["eps_start"]
        if self.steps > self.rl_config["eps_last_frame"]:
            return self.rl_config["eps_end"]
        dec_duration = self.rl_config["eps_last_frame"] - self.rl_config["eps_first_frame"]
        delta_step = self.steps - self.rl_config["eps_first_frame"]
        return self.rl_config["eps_start"] - (delta_step / dec_duration) * (
                self.rl_config["eps_start"] - self.rl_config["eps_end"])

    def on_train_epoch_end(self):
        super().on_train_epoch_end()

    @torch.no_grad()
    def update_target(self):
        if self.steps % self.rl_config["sync_rate"] == 0:
            self.target_mover.load_state_dict(self.mover.state_dict())

    def experience_state(self):
        self.requires_grad_(False)
        self.eval()

    def replay_state(self):
        self.mover.train()
        self.mover.requires_grad_(True)
        self.target_mover.eval()
        self.target_mover.requires_grad_(False)
        self.model.requires_grad_(False)
        self.model.eval()

    def warmup_dataloader(self, data, minimum_buffer_size, num_workers):
        return create_dataloader(data.data, minimum_buffer_size + data.batch_size, data.batch_size,
            num_workers=num_workers, drop_last=False, )


class QLMoverTrainSimul(QLMoverTrainOrd):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False
        self.train_eps = kwargs["train_eps"] if "train_eps" in kwargs else False

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        exp_batch, replay_batch, pred_batch = batch
        eps = self.get_epsilon()

        rl_opt, pred_opt = self.optimizers()
        self.pred_state()
        if self.train_eps:
            score, loss = self.compute_results(pred_batch, batch_idx, "train", eps=eps)
        else:
            score, loss = self.compute_results(pred_batch, batch_idx, "train")
        pred_opt.zero_grad()
        self.manual_backward(loss)
        pred_opt.step()
        self.experience_state()
        loss = self.exp_replay(exp_batch, replay_batch, eps)
        rl_opt.zero_grad()
        self.manual_backward(loss)
        rl_opt.step()
        self.log(osp.join(self.name, "train/rl_loss"), loss if loss is not None else 0, prog_bar=True,
            batch_size=replay_batch.g.batch_size, )
        self.log(osp.join(self.name, "eps"), eps, batch_size=replay_batch.g.batch_size, )
        self.steps += 1
        self.update_target()
        return loss

    def pred_state(self):
        self.mover.eval()
        self.mover.requires_grad_(False)
        self.target_mover.eval()
        self.target_mover.requires_grad_(False)
        self.model.train()
        self.model.requires_grad_(True)
