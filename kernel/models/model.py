import torch
import dgl
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch import nn
from torch.nn import functional as F
from gp.nn.models.util_model import MLP
from dgl.dataloading.dataloader import GraphCollator
from gp.nn.models.GNN import MultiLayerMessagePassing
from dgl.nn.pytorch.conv import GINEConv
from utils import gen_cumlabel


class LabelEncoder(nn.Module):
    """
    Transform a one-hot encoded label to a continuous feature
    """

    def __init__(self, label_dim, out_dim):
        super().__init__()
        self.label_dim = label_dim
        self.out_dim = out_dim
        self.feature_transform = MLP([self.label_dim, out_dim])
        self.label_norm = nn.BatchNorm1d(self.out_dim)

    def forward(self, labels):
        # labels = F.normalize(labels.to(torch.float), dim=-1)
        labels = labels.to(torch.float)
        return self.label_norm(self.feature_transform(labels))


class PlainLabelGNN(nn.Module):
    def __init__(self, emb_dim, label_encoder, task_gnn, pool="mean"):
        super().__init__()
        self.out_dim = emb_dim
        self.emb_dim = emb_dim
        self.task_gnn = task_gnn
        self.label_encoder = label_encoder
        self.collator = GraphCollator()
        self.gpool = MLP([self.emb_dim, self.emb_dim], plain_last=False)
        self.pool = pool

    def forward(self, inp_g, labels):
        """
        Generate embeddings based on labels. labels is either a matrix of one-hot labels
        or a 3-D tensor of one-hot labels. If a 3-D tensor, inp_g is repeated len(labels) times,
        so representations of the same graph with different labels are generated in parallel.

        For graph representation, self.pool="sum"/"mean" generates sum/mean pooling,
        self.pool="none" does not pool and output node representations.
        """
        if len(labels.size()) == 3:
            rep_size, nm_nodes, _ = labels.size()
            rep_size = len(labels)
            g = self.collator.collate([inp_g] * rep_size)
            labels = labels.view(rep_size * nm_nodes, -1)
        else:
            rep_size = 1
            g = inp_g
        m_label = gen_cumlabel(g, labels)
        label_emb = self.label_encoder(m_label)
        with g.local_scope():
            g.ndata["feat"] = torch.cat([g.ndata["feat"], label_emb], dim=-1)

            n_repr = self.task_gnn(g)
            inp_g.ndata["repr"] = self.gpool(n_repr.view(rep_size, inp_g.num_nodes(), -1).mean(dim=0))
            if self.pool == "sum":
                g_repr = dgl.sum_nodes(inp_g, "repr")
            elif self.pool == "mean":
                g_repr = dgl.mean_nodes(inp_g, "repr")
            elif self.pool == "none":
                g_repr = inp_g.ndata["repr"]
        return g_repr

    def get_out_dim(self):
        return self.emb_dim


"""
Following are dataset specific GNNs

"""


class MOLGNN(MultiLayerMessagePassing):
    def __init__(self, num_layers, inp_dim, out_dim, mol_emb_dim, edge_dim, atom_types, bond_types, drop_ratio=0,
                 JK="last", batch_norm=True, ):
        super().__init__(num_layers, inp_dim, out_dim, drop_ratio, JK, batch_norm)
        self.mol_emb_dim = mol_emb_dim
        self.edge_dim = edge_dim
        self.num_atom_types = atom_types
        self.num_bond_types = bond_types
        self.atom_encoder = torch.nn.Embedding(self.num_atom_types, self.mol_emb_dim)
        self.bond_encoder = torch.nn.ModuleList()
        self.encoded_inp_dim = inp_dim + self.mol_emb_dim

        self.build_layers()

    def build_input_layer(self):
        self.bond_encoder.append(torch.nn.Embedding(self.num_bond_types, self.encoded_inp_dim))
        return GINEConv(MLP([self.encoded_inp_dim, 2 * self.encoded_inp_dim, self.out_dim]), learn_eps=True, )

    def build_hidden_layer(self):
        self.bond_encoder.append(torch.nn.Embedding(self.num_bond_types, self.out_dim))
        return GINEConv(MLP([self.out_dim, 2 * self.out_dim, self.out_dim]))

    def layer_forward(self, layer, message):
        e_feat = self.bond_encoder[layer](message["bond"])

        return self.conv[layer](message["g"], message["h"], e_feat)

    def build_message_from_input(self, g):
        h = self.atom_encoder(g.ndata["atom_feat"])
        return {"g": g, "h": torch.cat([g.ndata["feat"], h], dim=-1, ), "bond": g.edata["bond_feat"], }

    def build_message_from_output(self, g, output):
        return {"g": g, "h": output, "bond": g.edata["bond_feat"], }


class MOLGNNQM(MOLGNN):
    def __init__(self, num_layers, inp_dim, out_dim, mol_emb_dim, edge_dim, drop_ratio=0, JK="last", batch_norm=True, ):
        super().__init__(num_layers, inp_dim + 11, out_dim, mol_emb_dim, edge_dim, 1000, 4, drop_ratio, JK,
                         batch_norm, )

    def layer_forward(self, layer, message):
        e_feat = self.bond_encoder[layer](message["bond"])

        return self.conv[layer](message["g"], message["h"], e_feat)

    def build_message_from_input(self, g):
        h = self.atom_encoder(g.ndata["atom_type"])
        return {"g": g, "h": torch.cat([g.ndata["feat"], h, g.ndata["atom_feat"]], dim=-1, ),
                "bond": g.edata["bond_feat"], }


class OGBMOLGNN(MultiLayerMessagePassing):
    def __init__(self, num_layers, inp_dim, out_dim, mol_emb_dim, edge_dim, drop_ratio=0, JK="last", batch_norm=True, ):
        super().__init__(num_layers, inp_dim, out_dim, drop_ratio, JK, batch_norm)
        self.mol_emb_dim = mol_emb_dim
        self.edge_dim = edge_dim

        self.atom_encoder = AtomEncoder(mol_emb_dim)
        self.bond_encoder = torch.nn.ModuleList()
        self.encoded_inp_dim = inp_dim + mol_emb_dim

        self.build_layers()

    def build_input_layer(self):
        self.bond_encoder.append(BondEncoder(self.encoded_inp_dim))
        return GINEConv(MLP([self.encoded_inp_dim, 2 * self.encoded_inp_dim, self.out_dim]), learn_eps=True, )

    def build_hidden_layer(self):
        self.bond_encoder.append(BondEncoder(self.out_dim))
        return GINEConv(MLP([self.out_dim, 2 * self.out_dim, self.out_dim]))

    def layer_forward(self, layer, message):
        e_feat = self.bond_encoder[layer](message["bond"])

        return self.conv[layer](message["g"], message["h"], e_feat)

    def build_message_from_input(self, g):
        return {"g": g, "h": torch.cat([g.ndata["feat"], self.atom_encoder(g.ndata["atom_feat"])], dim=-1, ),
                "bond": g.edata["bond_feat"], }

    def build_message_from_output(self, g, output):
        return {"g": g, "h": output, "bond": g.edata["bond_feat"], }
