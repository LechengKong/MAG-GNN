import torch
import os.path as osp
import networkx as nx
import numpy as np
import pickle as pkl
import os
import pickle
import torch_geometric as pyg
from ogb.graphproppred import DglGraphPropPredDataset

import dataset

from tqdm import tqdm
from gp.utils.graph import construct_dgl_graph_from_edges
from gp.utils.io import load_exp_dataset_dgl

from gp.utils.utils import (k_fold2_split, k_fold_ind, )

from dataset import (GraphLabelDataset, GraphReplayBuffer, ReplayDataset, )
from utils import to_dgl

"""
The following functions starting with 'prepare_' contain code to generate graphs and labels for training,
as well as a list of meta information. The return object is in the form of dictionary.
"""


def prepare_srg(params, data_folder_path, data_name):
    nx_graphs = nx.read_graph6(osp.join(data_folder_path, "sr251256.g6"))

    graphs = []
    label = []
    for j, g in enumerate(nx_graphs):
        edges = np.array(list(g.edges))
        for i in range(10):
            g = construct_dgl_graph_from_edges(edges[:, 0], edges[:, 1], n_entities=25, inverse_edge=True)
            g.ndata["feat"] = torch.ones((25, 0))
            graphs.append(g)
            label.append(j)
    inp_dim = 0
    num_class = 15
    return {"graphs": graphs, "labels": label, "inp_dim": inp_dim, "num_class": num_class, }


def prepare_brec(params, data_folder_path, data_name):
    g6_graph = np.load(osp.join(data_folder_path, params.brec_type + ".npy")).reshape(-1, 2)[params.brec_ind]
    nx_graph = [bytes(g, "utf-8") if isinstance(g, str) else g for g in g6_graph]
    nx_graph = [nx.from_graph6_bytes(g) for g in nx_graph]
    graphs = []
    label = []
    for j, g in enumerate(nx_graph):
        edges = np.array(list(g.edges))
        for i in range(10):
            g = construct_dgl_graph_from_edges(edges[:, 0], edges[:, 1], n_entities=g.number_of_nodes(),
                                               inverse_edge=True)
            g.ndata["feat"] = torch.ones((g.number_of_nodes(), 0))
            graphs.append(g)
            label.append(j)
    inp_dim = 0
    num_class = 2
    return {"graphs": graphs, "labels": label, "inp_dim": inp_dim, "num_class": num_class, }


def prepare_srg_test_cap(data_folder_path, data_name):
    nx_graphs = nx.read_graph6(osp.join(data_folder_path, "sr251256.g6"))

    graphs = []
    label = []
    r_label = []
    for j, g in enumerate(nx_graphs):
        edges = np.array(list(g.edges))
        for i in range(25):
            for k in range(25):
                g = construct_dgl_graph_from_edges(edges[:, 0], edges[:, 1], n_entities=25, inverse_edge=True, )
                g.ndata["feat"] = torch.ones((25, 1))
                graphs.append(g)
                label.append(j)
                r_label.append([i, k])
    inp_dim = 1
    num_class = 15
    return {"graphs": graphs, "labels": label, "inp_dim": inp_dim, "num_class": num_class, "r_label": r_label, }


def prepare_csl(params, data_folder_path, data_name):
    with open(osp.join(data_folder_path, "graphs_Kary_Deterministic_Graphs.pkl"), "rb", ) as file:
        d = pkl.load(file)

    # print((d[0]-d[0].T).sum())
    graphs = []
    for g in d:
        row = g.row
        col = g.col
        goodind = row != col
        row = row[goodind]
        col = col[goodind]
        g = construct_dgl_graph_from_edges(row, col, n_entities=g.shape[0])
        g.ndata["feat"] = torch.ones([g.num_nodes(), 0])
        graphs.append(g)
    label = torch.load(osp.join(data_folder_path, "y_Kary_Deterministic_Graphs.pt")).numpy()
    inp_dim = 0
    num_class = int(label.max() + 1)
    return {"graphs": graphs, "labels": label, "inp_dim": inp_dim, "num_class": num_class, }


def prepare_exp(params, data_folder_path, data_name):
    graphs, label = load_exp_dataset_dgl(osp.join(data_folder_path, "GRAPHSAT.txt"))
    inp_dim = 2
    num_class = 2
    return {"graphs": graphs, "labels": label, "inp_dim": inp_dim, "num_class": num_class, }


def prepare_zinc(params, data_folder_path, data_name):
    splits = ["train", "test", "val"]
    datasets = {}
    for s in splits:
        data = pyg.datasets.ZINC(data_folder_path, subset=True, split=s)
        labels = data.y
        graphs = []
        for i in tqdm(range(len(data))):
            g = to_dgl(data[i])
            g.ndata["feat"] = torch.ones([g.num_nodes(), 0])
            g.ndata["atom_feat"] = g.ndata["x"].view(-1)
            g.edata["bond_feat"] = g.edata["edge_attr"]
            graphs.append(g)
        labels = {"labels": labels}
        labels = labels["labels"]

        datasets[s] = [graphs, labels]

    inp_dim = 0
    num_class = 1
    return {"datasets": datasets, "inp_dim": inp_dim, "num_class": num_class, "node_type": 28, "edge_type": 4, }


def prepare_zinc_full(params, data_folder_path, data_name):
    splits = ["train", "test", "val"]
    datasets = {}
    for s in splits:
        data = pyg.datasets.ZINC(data_folder_path, split=s)
        labels = data.y
        graphs = []
        for i in tqdm(range(len(data))):
            g = to_dgl(data[i])
            g.ndata["feat"] = torch.ones([g.num_nodes(), 0])
            g.ndata["atom_feat"] = g.ndata["x"].view(-1)
            g.edata["bond_feat"] = g.edata["edge_attr"]
            graphs.append(g)
        labels = {"labels": labels}
        labels = labels["labels"]

        datasets[s] = [graphs, labels]

    inp_dim = 0
    num_class = 1
    return {"datasets": datasets, "inp_dim": inp_dim, "num_class": num_class, "node_type": 28, "edge_type": 4, }


def prepare_syn_cycle(params, data_folder_path, data_name):
    import scipy

    dt = scipy.io.loadmat(osp.join(data_folder_path, "data.mat"))
    adj = dt["A"][0]
    label = dt["F"][0]
    d_list = ["train", "val", "test"]
    datasets = {}
    if os.path.exists(osp.join(data_folder_path, "saved.pkl")):
        with open(osp.join(data_folder_path, "saved.pkl"), "rb") as f:
            datasets = pickle.load(f)
    else:
        for k in d_list:
            idx = dt[k + "_idx"][0]
            g_list = []
            l_list = []
            pbar = tqdm(idx)
            for i in pbar:
                edges = adj[i].nonzero()
                n_nodes = len(adj[i])
                graph = construct_dgl_graph_from_edges(edges[0], edges[1], n_nodes)
                graph.ndata["feat"] = torch.ones([graph.num_nodes(), 0])
                glabel = label[i]
                g_list.append(graph)
                l_list.append(glabel)
            l_list_cat = np.concatenate(l_list)
            if k == "train":
                label_mean = np.mean(l_list_cat, axis=0)
                label_std = np.std(l_list_cat, axis=0)
            for i in range(len(l_list)):
                l_list[i] = (l_list[i] - label_mean) / label_std
            datasets[k] = [g_list, l_list]

    for k in datasets:
        for i in range(len(datasets[k][1])):
            datasets[k][0][i].ndata["tlabel"] = torch.tensor(datasets[k][1][i][:, params.data_label])
            datasets[k][1][i] = 0
    data = {"inp_dim": 0, "num_class": 1, "datasets": datasets, }

    return data


def prepare_qm(params, data_folder_path, data_name):
    name = params.data_label
    names = ["mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "U0", "U", "H", "G", "Cv", ]
    HAR2EV = 27.2113825435
    KCALMOL2EV = 0.04336414
    conversion = torch.tensor(
        [1.0, 1.0, HAR2EV, HAR2EV, HAR2EV, 1.0, HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV, 1.0, KCALMOL2EV, KCALMOL2EV,
         KCALMOL2EV, KCALMOL2EV, 1.0, 1.0, 1.0, ])
    n2i = {n: i for i, n in enumerate(names)}
    data = pyg.datasets.QM9(data_folder_path)
    labels = data.y / conversion
    graphs = []
    for i in tqdm(range(len(data))):
        g = to_dgl(data[i])
        g.ndata["feat"] = torch.ones([g.num_nodes(), 0])
        g.ndata["atom_feat"] = g.ndata["x"]
        g.edata["bond_feat"] = torch.argmax(g.edata["edge_attr"], dim=-1)
        g.ndata["atom_type"] = g.ndata["z"]
        graphs.append(g)
    labels = {"labels": labels}
    labels = labels["labels"]
    labels = labels[:, n2i[name]]

    data = {"graphs": graphs, "labels": labels, "inp_dim": 0, "edge_dim": 4, "num_class": 1, "node_dim": 11, }
    return data


def prepare_hiv(params, data_folder_path, data_name):
    data = DglGraphPropPredDataset(name=data_name, root=data_folder_path)
    split_idx = data.get_idx_split()
    train_ind = split_idx["train"]
    test_ind = split_idx["test"]
    val_ind = split_idx["valid"]
    datasets = {}
    train_graph = [data[i][0] for i in train_ind]
    test_graph = [data[i][0] for i in test_ind]
    val_graph = [data[i][0] for i in val_ind]

    train_label = [data[i][1].numpy() for i in train_ind]
    test_label = [data[i][1].numpy() for i in test_ind]
    val_label = [data[i][1].numpy() for i in val_ind]
    datasets["train"] = [train_graph, train_label]
    datasets["test"] = [test_graph, test_label]
    datasets["val"] = [val_graph, val_label]
    splits = ["train", "test", "val"]
    for k in splits:
        for g in datasets[k][0]:
            g.ndata["atom_feat"] = g.ndata["feat"]
            g.edata["bond_feat"] = g.edata["feat"]
            g.ndata["feat"] = torch.ones([g.num_nodes(), 0])

    data = {"num_class": 2, "inp_dim": 0, "datasets": datasets, }
    return data


def prepare_data(params, data_path, data_name, *args):
    if data_name not in available_data_prepare_func:
        raise NotImplementedError(data_name + " not available")
    data_folder_path = osp.join(data_path, data_name)
    return available_data_prepare_func[data_name](params, data_folder_path, data_name, *args)


available_data_prepare_func = {"srg": prepare_srg, "csl": prepare_csl, "exp": prepare_exp, "zinc": prepare_zinc,
                               "syn_count": prepare_syn_cycle, "ogbg-molhiv": prepare_hiv, "qm": prepare_qm,
                               "zinc_full": prepare_zinc_full, "brec": prepare_brec, }


"""
Following functions are data transformation functions that takes in a data dictionary generated by 'prepare_' functions
and a params parameter dictionary to generate torch datasets for training.
"""


def graph_data_k_fold_split(data, params):
    """
    This function split "graph" and "labels" to train/valid/test set by fold parameter in params, in a
    stratified manner whenever possible. It then builds corresponding datasets.
    """
    labels = data["labels"]
    graphs = data["graphs"]
    folds = k_fold_ind(labels, fold=params.fold)
    splits = k_fold2_split(folds, len(labels))
    s = splits[0]
    if hasattr(params, "data_norm") and params.data_norm:
        arr = np.array([labels[i] for i in s[0]])
        mean = arr.mean()
        std = arr.std()
        labels = np.array(labels)
        labels = (labels - mean) / std
        data["mean"] = mean
        data["std"] = std

    train = GraphLabelDataset([graphs[i] for i in s[0]], [labels[i] for i in s[0]])
    test = GraphLabelDataset([graphs[i] for i in s[1]], [labels[i] for i in s[1]])
    val = GraphLabelDataset([graphs[i] for i in s[2]], [labels[i] for i in s[2]])

    data["num_batches"] = int(len(train) / params.batch_size)

    data["rl_train"] = train
    data["train"] = train
    data["test"] = test
    data["valid"] = val


def graph_splitted_data(data, params):
    """
    This function assumes data contains "dataset" with splitted graphs and labels.
    """
    train = GraphLabelDataset(data["datasets"]["train"][0], data["datasets"]["train"][1])
    test = GraphLabelDataset(data["datasets"]["test"][0], data["datasets"]["test"][1])
    val = GraphLabelDataset(data["datasets"]["val"][0], data["datasets"]["val"][1])
    data["num_batches"] = int(len(train) / params.batch_size)
    data["train"] = train
    data["rl_train"] = train
    data["test"] = test
    data["valid"] = val


def add_replay_data(data, params):
    """
    This function builds replay buffer/ replay dataset based on the "rl_train" dataset
    """
    replay_buffer = GraphReplayBuffer(params.memory_capacity)
    data["replay_buffer"] = replay_buffer
    if params.replay_type == "hist":
        replay_type = dataset.ReplayBatchHist
    elif params.replay_type == "plain":
        replay_type = dataset.ReplayBatch
    elif params.replay_type == "sm":
        replay_type = dataset.ReplayBatchSM

    replay_train = ReplayDataset(data["rl_train"].graphs, replay_buffer, params.replay_size, replay_type)
    data["replay_train"] = replay_train


def graph_splitted_half(data, params):
    ind = np.random.permutation(len(data["datasets"]["train"][0]))
    half_ind = int(len(data["datasets"]["train"][0]) / 2)
    train_one_ind = ind[:half_ind]
    train_two_ind = ind[half_ind:]
    train = GraphLabelDataset(data["datasets"]["train"][0], data["datasets"]["train"][1])
    train_one = GraphLabelDataset([data["datasets"]["train"][0][i] for i in train_one_ind],
                                  [data["datasets"]["train"][1][i] for i in train_one_ind], )
    train_two = GraphLabelDataset([data["datasets"]["train"][0][i] for i in train_two_ind],
                                  [data["datasets"]["train"][1][i] for i in train_two_ind], )
    test = GraphLabelDataset(data["datasets"]["test"][0], data["datasets"]["test"][1])
    val = GraphLabelDataset(data["datasets"]["val"][0], data["datasets"]["val"][1])
    data["num_batches"] = int(len(train_one) / params.batch_size)
    data["train"] = train
    data["train_0"] = train_one
    data["train_1"] = train_two
    data["test"] = test
    data["valid"] = val


def build_train_half(data, params):
    data["train"] = data["train_0"]
    data["rl_train"] = data["train_1"]
