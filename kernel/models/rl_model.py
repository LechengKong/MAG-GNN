import torch
import dgl
import torch.nn.functional as F
from abc import abstractmethod, ABCMeta
from gp.nn.models.util_model import MLP
from utils import var_size_repeat, gen_cumlabel, random_label
from torch_scatter import scatter_max


class NMRLModel(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self, gnn, num_piece, contexter=None):
        super().__init__()
        self.gnn = gnn
        self.num_piece = num_piece
        self.contexter = contexter
        self.collator = dgl.dataloading.dataloader.GraphCollator()

    @abstractmethod
    def forward(self, g, labels, *args):
        pass

    @abstractmethod
    def compute_q_val(self, g, labels, *args):
        pass

    @abstractmethod
    def return_action_q(self, g, labels, *args):
        pass

    @abstractmethod
    def state_transition(self, old_state, action):
        pass

    @abstractmethod
    def eps_action(self, action, eps, g):
        pass

    def compute_NM_g_repr(self, g, labels):
        """
        Compute graph representations based on input graph and set of labels

        Returns:
            all_repr: node representations
            g_nm_repr: graph representations under one node marking.
            g_repr: pooled graph representation over all current node markings.
        """
        nm_labels = len(labels)
        flat_labels = labels.view(nm_labels * g.batch_size, -1)
        bg = self.collator.collate([g] * nm_labels)
        # m_label = gen_cumlabel(bg, flat_labels)
        all_repr = self.gnn(bg, flat_labels)
        with bg.local_scope():
            bg.ndata["repr"] = all_repr
            g_nm_repr = dgl.sum_nodes(bg, "repr")
            g_repr = g_nm_repr.view(nm_labels, g.batch_size, -1).mean(dim=0)
        return all_repr, g_nm_repr, g_repr

    def gen_context(self, g_repr, *args):
        if self.contexter is not None:
            return self.contexter.gen_context(g_repr, *args)
        return g_repr


class PieceMover(NMRLModel, metaclass=ABCMeta):
    @abstractmethod
    def compute_max_move_from_q_val(self, q_val, g):
        pass

    def forward(self, g, labels, *args):
        q_val, context = self.compute_q_val(g, labels, *args)
        max_q_val, dest_ind, src_nm = self.compute_max_move_from_q_val(q_val, g)
        return max_q_val, (dest_ind, src_nm), context

    def state_transition(self, old_state, action):
        nm_labels, g_size, _ = old_state.size()
        label = old_state.view(nm_labels * g_size, -1)
        dest_ind = action[0].view(-1)
        src_nm = action[1].view(-1)
        new_label = torch.clone(label)
        new_label[torch.arange(len(new_label)), src_nm] = dest_ind
        new_label = new_label.view(old_state.size())
        return new_label

    def eps_action(self, action, eps, g):
        """
        Select random action with eps probability
        """
        dest_ind = action[0].view(-1)
        src_nm = action[1].view(-1)
        num_graphs = g.batch_size * len(action[0])
        rand_select = random_label(1, num_graphs,
            torch.zeros(num_graphs, dtype=torch.long, device=g.device) + self.num_piece, ).view(-1)
        rand_label = random_label(1, num_graphs, g.batch_num_nodes().repeat(len(action[0])), ).view(-1)
        rand_prob = torch.rand(num_graphs) < eps
        src_nm[rand_prob] = rand_select[rand_prob]
        dest_ind[rand_prob] = rand_label[rand_prob]
        return dest_ind.view(action[0].size()), src_nm.view(action[1].size())


class PieceMoverMLP(PieceMover):
    def __init__(self, *args):
        super().__init__(*args)
        self.move_mlp = MLP([self.gnn.out_dim * 3, self.gnn.out_dim * 4, self.gnn.out_dim * 2, self.num_piece, ])

    def compute_max_move_from_q_val(self, q_val, g):
        """
        Compute most rewarding movement for each node marking.
        """
        nm_labels, nm_nodes, _ = q_val.size()
        g_ind = torch.arange(g.batch_size * nm_labels, device=g.device).repeat_interleave(
            g.batch_num_nodes().repeat(nm_labels))
        q_val_flat = q_val.view(nm_labels * nm_nodes, -1)
        max_q_val, max_ind = torch.max(q_val_flat, dim=-1)
        max_val, move_ind = scatter_max(max_q_val, g_ind, dim_size=g.batch_size * nm_labels)
        max_ind = max_ind[move_ind]
        move_ind = move_ind - torch.cumsum(torch.cat(
            [torch.tensor([0], dtype=torch.long, device=g.device)] + [g.batch_num_nodes()] * (nm_labels - 1) + [
                g.batch_num_nodes()[:-1]], dim=0, ), dim=0, )
        return (max_val.view(nm_labels, -1), move_ind.view(nm_labels, -1), max_ind.view(nm_labels, -1),)

    def compute_q_val(self, g, labels, *args):
        """
        Given the representation, compute q values.
        """
        all_repr, g_nm_repr, g_repr = self.compute_NM_g_repr(g, labels)

        context = self.gen_context(g_repr, g, *args)
        g_nm_repr_expand = g_nm_repr.repeat_interleave(g.batch_num_nodes().repeat(len(labels)), dim=0)
        g_repr_expand = g_repr.repeat_interleave(g.batch_num_nodes(), dim=0).repeat(len(labels), 1)
        all_repr = F.normalize(all_repr, dim=-1)
        move_repr = torch.cat([all_repr, g_nm_repr_expand, g_repr_expand], dim=-1, )
        q_val = self.move_mlp(move_repr).view(len(labels), g.num_nodes(), -1)
        return q_val, context

    def return_action_q(self, g, labels, select_node, move_node, *args):
        """
        Compute the q-value of a given action
        """
        q_val, context = self.compute_q_val(g, labels, *args)
        nm_labels, nm_nodes, _ = q_val.size()
        offset = torch.cumsum(torch.cat(
            [torch.tensor([0], dtype=torch.long, device=g.device)] + [g.batch_num_nodes()] * (nm_labels - 1) + [
                g.batch_num_nodes()[:-1]], dim=0, ), dim=0, )
        move_node = move_node.view(-1) + offset

        return (q_val.view(nm_labels * nm_nodes, -1)[move_node, select_node.view(-1)].view(nm_labels, -1), context,)
