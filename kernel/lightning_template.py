import torch
import os.path as osp
from abc import ABCMeta
from torch.utils.data import DataLoader, RandomSampler
from pytorch_lightning import LightningModule
from gp.lightning.module_template import BaseTemplate


class GraphPredTemplate(BaseTemplate):
    def forward(self, batch):
        graph = batch.g
        g_repr = self.model["task_predictor"](graph, batch)
        x = self.model["classifier"](g_repr)
        return x


class QLTemplate(BaseTemplate, metaclass=ABCMeta):
    def __init__(
        self,
        exp_config,
        environment,
        mover,
        agent,
        k,
        eval_kit,
        name="rl",
    ):
        super().__init__(
            exp_config,
            environment,
            eval_kit,
            name,
        )
        self.mover = mover
        self.k = k
        self.agent = agent

    def forward(self, batch, eps=None):
        # print(self.mover.no_hist_prior)
        g = batch.g
        labels = self.agent.search(
            self.mover,
            self.model,
            g,
            self.k,
            eps,
            labels=None,
            batch=batch,
        )

        pred = self.model["task_predictor"](g, labels)
        pred = self.model["classifier"](pred)
        return pred

    def on_train_epoch_start(self):
        self.model.train()
        self.model.requires_grad_(True)
        self.mover.eval()
        self.mover.requires_grad_(False)
