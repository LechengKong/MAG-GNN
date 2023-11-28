import torch
from gp.nn.models.dgl import DGLGIN
from gp.nn.models.util_model import MLP
from kernel.models.rl_model import PieceMoverMLP
from kernel.agent import MultiSMAgent

from kernel.models.model import (
    PlainLabelGNN,
    MOLGNN,
    # OGBMOLGNN,
    MOLGNNQM,
    LabelEncoder, OGBMOLGNN,
)


def build_cl(label_encoder, gnn, emb_dim, out_dim, pool="mean"):
    graph_encoder = PlainLabelGNN(emb_dim, label_encoder, gnn, pool=pool)

    classifier = MLP(
        [
            graph_encoder.get_out_dim(),
            graph_encoder.get_out_dim() * 2,
            graph_encoder.get_out_dim() * 2,
            out_dim,
        ],
    )

    return torch.nn.ModuleDict(
        {
            "graph_encoder": graph_encoder,
            "task_predictor": graph_encoder,
            "classifier": classifier,
        }
    )


def prepare_no_feat_graph(params, data):
    label_encoder = LabelEncoder(params.num_piece, params.emb_dim)
    pred_gnn = DGLGIN(
        params.num_layers,
        params.emb_dim + data["inp_dim"],
        params.emb_dim,
        batch_norm=True,
        JK=params.JK,
    )

    return build_cl(label_encoder, pred_gnn, params.emb_dim, data["num_class"])


def prepare_no_feat_node(params, data):
    label_encoder = LabelEncoder(params.num_piece, params.emb_dim)
    pred_gnn = DGLGIN(
        params.num_layers,
        params.emb_dim + data["inp_dim"],
        params.emb_dim,
        batch_norm=True,
        JK=params.JK,
    )

    return build_cl(label_encoder, pred_gnn, params.emb_dim, data["num_class"], "none")


def prepare_ZINC(params, data):
    label_encoder = LabelEncoder(params.num_piece, params.emb_dim)
    pred_gnn = MOLGNN(
        params.num_layers,
        params.emb_dim,
        params.emb_dim,
        params.emb_dim,
        0,
        data["node_type"],
        data["edge_type"],
        JK=params.JK,
    )

    return build_cl(label_encoder, pred_gnn, params.emb_dim, data["num_class"])


def prepare_OGB(params, data):
    label_encoder = LabelEncoder(params.num_piece, params.emb_dim)
    pred_gnn = OGBMOLGNN(
        params.num_layers,
        params.emb_dim,
        params.emb_dim,
        params.emb_dim,
        0,
        JK=params.JK,
    )

    return build_cl(label_encoder, pred_gnn, params.emb_dim, data["num_class"])


def prepare_QM(params, data):
    label_encoder = LabelEncoder(params.num_piece, params.emb_dim)
    pred_gnn = MOLGNNQM(
        params.num_layers,
        params.emb_dim,
        params.emb_dim,
        8,
        0,
        JK=params.JK,
    )

    return build_cl(label_encoder, pred_gnn, params.emb_dim, data["num_class"])


def prepare_model_rl(params, data, num_layers=None):
    mover_type = PieceMoverMLP
    if num_layers is None:
        num_layers = params.num_layers
    label_encoder = LabelEncoder(params.num_piece, params.emb_dim)
    if hasattr(params, "node_only_agent") and params.node_only_agent:
        move_gnn = DGLGIN(
            num_layers,
            params.emb_dim + data["inp_dim"],
            params.emb_dim,
            batch_norm=True,
        )
    else:
        if params.train_data_set in ["srg", "syn_count", "brec"]:
            move_gnn = DGLGIN(
                num_layers,
                params.emb_dim + data["inp_dim"],
                params.emb_dim,
                batch_norm=True,
            )
        if params.train_data_set in ["zinc", "zinc_full"]:
            move_gnn = MOLGNN(
                params.num_layers,
                params.emb_dim,
                params.emb_dim,
                params.emb_dim,
                0,
                data["node_type"],
                data["edge_type"],
                JK=params.JK,
            )
        if params.train_data_set in ["qm"]:
            move_gnn = MOLGNNQM(
                params.num_layers,
                params.emb_dim,
                params.emb_dim,
                8,
                0,
                JK=params.JK,
            )
        if params.train_data_set in ["ogbg-molhiv"]:
            move_gnn = OGBMOLGNN(
                params.num_layers,
                params.emb_dim,
                params.emb_dim,
                params.emb_dim,
                0,
                JK=params.JK,
            )
    move_gnn = PlainLabelGNN(params.emb_dim, label_encoder, move_gnn, pool="none")

    histcombine = None

    move_q_network = mover_type(move_gnn, params.num_piece, histcombine)

    return move_q_network


def prepare_agent(params, data, rl_models, models, eval_func):
    move_agent = MultiSMAgent(
        params.num_piece,
        data["replay_buffer"],
        eval_func=eval_func,
        experience_step=params.exp_steps,
        num_nm=params.num_nm,
        post_method=params.post_method,
        state_emb=params.state_emb,
        ens=params.ens,
    )

    return move_agent
