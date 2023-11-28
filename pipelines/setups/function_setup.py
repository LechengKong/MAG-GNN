import inspect

import torch.nn

import pipelines.setups.lightning_prepare
import pipelines.setups.metric_prepare
import pipelines.setups.model_prepare
import data_prep

from utils import eval_cls, eval_reg, eval_reg_node, convert_yaml_params, load_environment, load_rl
from pipelines.setups.model_prepare import prepare_model_rl

lightning_func_name = dict(
    inspect.getmembers(pipelines.setups.lightning_prepare, inspect.isfunction)
)
model_func_name = dict(
    inspect.getmembers(pipelines.setups.model_prepare, inspect.isfunction)
)
data_prep_name = dict(inspect.getmembers(data_prep, inspect.isfunction))
eval_func_name = dict(inspect.getmembers(pipelines.setups.metric_prepare, inspect.isfunction))


def make_functions_from_params(params):
    make_model(params)
    make_data(params)
    make_eval(params)
    make_loss(params)
    make_func(params)


def make_model(params):
    params.prepare_model = model_func_name[params.prepare_model]


def make_data(params):
    params.data_trans = [data_prep_name[n] for n in params.data_trans]


def make_eval(params):
    if params.agent_eval == "cls":
        params.agent_eval = eval_cls
    elif params.agent_eval == "reg":
        params.agent_eval = eval_reg
    elif params.agent_eval == "reg_node":
        params.agent_eval = eval_reg_node


def make_loss(params):
    if params.loss == "mae":
        params.loss = torch.nn.L1Loss()
    elif params.loss == "ce":
        params.loss = torch.nn.CrossEntropyLoss()


def make_func(params):
    params.eval_func = eval_func_name[params.eval_func]
    params.loss_func = eval_func_name[params.loss_func]


def safe_load_create_env(params, data):
    if hasattr(params, "env_load"):
        if params.env_load is None:
            raise FileNotFoundError("env path not specified")
        env_params = convert_yaml_params(params.env_load[0])
        make_functions_from_params(env_params)
        env_state = torch.load(params.env_load[1])["state_dict"]
        torch_models = env_params.prepare_model(env_params, data)
        load_environment(torch_models, env_state)
    else:
        env_params = params
        torch_models = env_params.prepare_model(env_params, data)
    return torch_models

def safe_load_create_mover(params, data):
    if hasattr(params, "mover_load"):
        if params.mover_load is None:
            raise FileNotFoundError("mover path not specified")
        env_params = convert_yaml_params(params.mover_load[0])
        make_functions_from_params(env_params)
        env_state = torch.load(params.mover_load[1])["state_dict"]
        rl_models = prepare_model_rl(env_params, data)
        rl_target_models = prepare_model_rl(env_params, data)
        load_rl(rl_models, env_state)
    else:
        env_params = params
        rl_models = prepare_model_rl(env_params, data)
        rl_target_models = prepare_model_rl(env_params, data)
    return rl_models, rl_target_models