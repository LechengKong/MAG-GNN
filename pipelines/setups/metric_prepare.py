from typing import Any

import torch

from gp.lightning.metric import EvalKit

from torchmetrics import Accuracy, AUROC, MeanAbsoluteError


def build_eval_kit(data, params, name, eval_train=True, std=None):
    if eval_train:
        eval_data = data["val"] + data["test"]
    else:
        eval_data = data['train'] + data["val"] + data["test"]
    val_state = [dt.state_name for dt in data["val"]]
    test_state = [dt.state_name for dt in data["test"]]
    eval_state = [dt.state_name for dt in eval_data]
    eval_metric = [dt.metric for dt in eval_data]
    eval_funcs = [dt.meta_data["eval_func"] for dt in eval_data]
    loss = params.loss
    loss_func = params.loss_func
    evlter = []
    for dt in eval_data:
        if dt.metric == "acc":
            evlter.append(Accuracy(task="multiclass", num_classes=dt.classes))
        elif dt.metric == "auc":
            evlter.append(AUROC(task="binary"))
        elif dt.metric == "mae":
            evlter.append(MeanAbsoluteError())
        elif dt.metric == "mae_std":
            evlter.append(MaeWithStd(std))
    metrics = EvalKit(
        eval_metric,
        evlter,
        loss,
        eval_funcs,
        loss_func,
        eval_mode=params.eval_mode,
        exp_prefix=name,
        eval_state=eval_state,
        val_monitor_state=val_state[0],
        test_monitor_state=test_state[0],
    )
    return metrics


def classification_func(func, output, batch):
    return func(output, batch.labels.view(-1).to(torch.long))


def last_col_auc(func, output, batch):
    return func(output[:, -1], batch.labels.view(-1).to(torch.long))


def regression_func(func, output, batch):
    return func(output.view(-1), batch.labels.view(-1))


def regression_func_node(func, output, batch):
    return func(output.view(-1), batch.g.ndata["tlabel"].view(-1))


class MaeWithStd(MeanAbsoluteError):
    def __init__(self, std):
        super().__init__()
        self.std = std

    def update(self, preds, targets):
        super().update(preds * self.std, targets * self.std)
