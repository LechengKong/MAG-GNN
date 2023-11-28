import torch
import os
import os.path as osp
import numpy as np
import yaml
import dgl
from torch_geometric.data import Data, HeteroData

from gp.utils.graph import shortest_dist_sparse_mult
from datetime import datetime

from types import SimpleNamespace


def setup_exp(params):
    if not osp.exists("./saved_exp"):
        os.mkdir("./saved_exp")

    curtime = datetime.now()
    exp_dir = osp.join("./saved_exp", str(curtime))
    os.mkdir(exp_dir)
    with open(osp.join(exp_dir, "command"), "w") as f:
        yaml.dump(params, f)
    params["exp_dir"] = exp_dir


"""
Following functions contains utility function that generates labels/marking
"""


def random_label(dimension, b_size, block_sizes):
    """
    Generate random indices from a fixed range specified by block_sizes.
    Args:
        dimension: length of each generated index set
        b_size: number of generated sets
        block_sizes: range of each set

    Returns: b_size * dimension tensor T, where

    0 <= T[i, j] <= block_sizes[i]  for all j

    """
    t = torch.rand(b_size, dimension).to(block_sizes.device)
    pos_ind = t * block_sizes[:, None]
    pos_ind = torch.floor(pos_ind).to(torch.long)
    return pos_ind


def gen_cumlabel(g, pos_ind):
    """
    Convert sets of indices to one-hot representation
    """
    N = g.batch_num_nodes()
    N_count = torch.cat(
        [torch.tensor([0], device=N.device), torch.cumsum(N, dim=-1)]
    )
    pos_ind_in_graph = pos_ind + N_count[:-1, None]
    ind_col = torch.arange(pos_ind.size()[1], device=pos_ind.device).repeat(
        len(pos_ind)
    )
    label = (
        torch.zeros(
            [g.num_nodes(), pos_ind.size()[1]],
            device=pos_ind.device,
            dtype=torch.long,
        )
        + 1
    )
    # print(pos_ind_in_graph.max())
    label[pos_ind_in_graph.flatten().to(torch.long), ind_col.flatten()] = -1
    return label


def gen_distlabel(g, pos_ind):
    N = g.batch_num_nodes()
    N_count = torch.cat(
        [torch.tensor([0], device=N.device), torch.cumsum(N, dim=-1)]
    )
    pos_ind_in_graph = pos_ind + N_count[:-1, None]

    adjmat = g.adj(scipy_fmt="csr")
    dist = shortest_dist_sparse_mult(
        adjmat, 2, pos_ind_in_graph.cpu().numpy().flatten()
    )
    dist = dist.T

    ind_col = (
        torch.arange(
            pos_ind.size()[1] * pos_ind.size()[0], device=pos_ind.device
        )
        .reshape(pos_ind.size())
        .repeat_interleave(N, dim=0)
        .cpu()
        .numpy()
    )
    label = dist[
        np.arange(len(dist)).reshape(-1, 1).repeat(pos_ind.size()[1], axis=1),
        ind_col,
    ]
    return torch.tensor(label, dtype=torch.float, device=pos_ind.device)


def nm_init(g, num_nm, num_piece):
    labels = []
    for i in range(num_nm):
        label = random_label(num_piece, g.batch_size, g.batch_num_nodes())
        labels.append(label)
    return torch.stack(labels, dim=0)


def var_size_repeat(size, chunks, repeats):
    a = torch.arange(size, device=chunks.device)
    s = torch.cat(
        [torch.tensor([0], device=chunks.device), torch.cumsum(chunks, dim=0)]
    )
    starts = a[torch.repeat_interleave(s[:-1], repeats)]
    if len(starts) == 0:
        return torch.tensor([])
    offset = torch.repeat_interleave(chunks, repeats)
    ends = starts + offset

    clens = torch.cumsum(offset, dim=0)
    ids = torch.ones(clens[-1], dtype=torch.long, device=chunks.device)
    ids[0] = starts[0]
    ids[clens[:-1]] = starts[1:] - ends[:-1] + 1
    out = torch.cumsum(ids, dim=0)
    return out


def load_yaml(dir):
    with open(dir, "r") as stream:
        return yaml.safe_load(stream)


"""
Following are score evaluation functions for different tasks.
"""

def eval_cls(score, batch):
    overall_score = torch.nn.functional.softmax(score, dim=-1)[
        torch.arange(len(score)), batch.labels.view(-1)
    ]
    return overall_score


def eval_reg(score, batch):
    res = torch.abs(score - batch.labels).view(-1)
    return -res


def eval_reg_node(score, batch):
    res = torch.abs(score.view(-1) - batch.g.ndata["tlabel"].view(-1))
    g = batch.g
    with g.local_scope():
        g.ndata["err"] = res
        err = dgl.sum_nodes(g, "err")
    return -err


"""
Logging utility functions
"""

def log_step(logger, metric, name, mean, std):
    logger.log_metrics(
        {
            osp.join(name, "test", metric) + "_mean": mean,
            osp.join(name, "test", metric) + "_std": std,
        },
    )


def log_mean_var(logger, metric, mean, std):
    logger.log_metrics(
        {
            metric + "_mean": mean,
            metric + "_std": std,
        },
    )


"""
Model IO utility functions
"""

def load_environment(current_model, state_dict):
    new_state_dict = {}
    for name, param in state_dict.items():
        if name.startswith("model"):
            new_state_dict[name[6:]] = param
    # print(current_model.state_dict())
    current_model.load_state_dict(new_state_dict)


def load_rl(current_model, state_dict):
    new_state_dict = {}
    for name, param in state_dict.items():
        if name.startswith("mover"):
            new_state_dict[name[6:]] = param
    current_model.load_state_dict(
        new_state_dict, strict=False
    )


def load_graph_encoder(current_model, state_dict):
    new_state_dict = {}
    for name, param in state_dict.items():
        if name.startswith("model.graph_encoder"):
            new_state_dict[name[6:]] = param
    current_model.load_state_dict(new_state_dict, strict=False)


def convert_yaml_params(params_path):
    load_params = load_yaml(params_path)
    load_params = SimpleNamespace(**load_params)
    return load_params


def to_dgl(data):
    """
    Convert a pytorch_geometric Data to dgl graph, copied from pytorch_geometric source code.
    """

    if isinstance(data, Data):
        if data.edge_index is not None:
            row, col = data.edge_index
        else:
            row, col, _ = data.adj_t.t().coo()

        g = dgl.graph((row, col))

        for attr in data.node_attrs():
            g.ndata[attr] = data[attr]
        for attr in data.edge_attrs():
            if attr in ["edge_index", "adj_t"]:
                continue
            g.edata[attr] = data[attr]

        return g

    if isinstance(data, HeteroData):
        data_dict = {}
        for edge_type, store in data.edge_items():
            if store.get("edge_index") is not None:
                row, col = store.edge_index
            else:
                row, col, _ = store["adj_t"].t().coo()

            data_dict[edge_type] = (row, col)

        g = dgl.heterograph(data_dict)

        for node_type, store in data.node_items():
            for attr, value in store.items():
                g.nodes[node_type].data[attr] = value

        for edge_type, store in data.edge_items():
            for attr, value in store.items():
                if attr in ["edge_index", "adj_t"]:
                    continue
                g.edges[edge_type].data[attr] = value

        return g

    raise ValueError(f"Invalid data type (got '{type(data)}')")
