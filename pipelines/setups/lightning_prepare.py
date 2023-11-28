from kernel.lightning_models import (
    GraphRandLabelPredTemplate,
    QLMoverTrainOrd,
    GraphFixedLabelPredTemplate,
    QLMoverTrainSimul,
)

from kernel.lightning_template import QLTemplate, GraphPredTemplate


def prepare_lightning_rand_NM(params,
                              models,
                              exp_config,
                              eval_kit,
                              name=""):
    graph_pred = GraphRandLabelPredTemplate(exp_config, models, eval_kit, params.num_nm, params.num_piece, name)
    return graph_pred


def prepare_lightning_rl(
        params,
        models,
        rl_models,
        target_rl_models,
        agent,
        exp_config,
        eval_kit,
        name="",
):
    if params.training_mode == "simul":
        rl_lightning_class = QLMoverTrainSimul
    else:
        rl_lightning_class = QLMoverTrainOrd
    rl_config = {"sync_rate": params.sync_rate, "eps_start": params.eps_start, "eps_end": params.eps_end,
                 "eps_first_frame": params.eps_first_frame, "eps_last_frame": params.eps_last_frame,
                 "gamma": params.gamma}
    graph_rl = rl_lightning_class(exp_config, models, rl_models, target_rl_models, agent, params.q_steps, eval_kit,
                                  name,
                                  rl_config, train_eps=params.train_eps)
    return graph_rl


def prepare_lightning_rl_pred(
        params,
        models,
        rl_models,
        agent,
        exp_config,
        eval_kit,
        name="",
):
    graph_rl_pred = QLTemplate(
        exp_config,
        models,
        rl_models,
        agent,
        params.q_steps,
        eval_kit,
        name,
    )
    return graph_rl_pred
