import torch
import dgl
import numpy as np
from collections import namedtuple
from utils import random_label, eval_cls, nm_init
from gp.utils.utils import SmartTimer
from dgl.dataloading.dataloader import GraphCollator


class MultiSMAgent(torch.nn.Module):
    Experience = namedtuple("Experience", ["g_index", "plabel", "nlabel", "select_node", "move_node", "reward", ], )

    def __init__(self, num_piece, buffer, experience_step=4, eval_func=eval_cls, num_nm=None, post_method="direct",
            state_emb="rl", ens=False, ):
        super().__init__()
        self.num_piece = num_piece
        self.experience_step = experience_step
        self.buffer = buffer
        self.eval_func = eval_func
        self.timer = SmartTimer(False)
        self.num_nm = num_nm
        self.post_method = post_method
        self.state_emb = state_emb
        self.ens = ens

    def sample_experience(self, size, dataset, replay_type):
        replays = self.buffer.sample(size)
        for i in range(len(replays)):
            replay_graph = dataset.graphs[int(replays[i][0])]
            replays[i] = replay_type(replay_graph, *map(np.array, replays[i][1:]))
        replay_batch = GraphCollator().collate(replays)
        return replay_batch

    def get_action(self, mover, label, g, epsilon, *args):
        max_q, action, context = mover(g, label, *args)
        if epsilon is not None:
            action = mover.eps_action(action, epsilon, g)
        new_label = mover.state_transition(label, action)
        return new_label, action, context

    def eval_label(self, g, label, environment):
        g_repr = environment["graph_encoder"](g, label)
        return g_repr

    def eval_state(self, g_repr, environment, batch):
        return self.eval_func(environment["classifier"](g_repr), batch)

    @torch.no_grad()
    def experience_fn(self, mover, environment, eps, batch, labels=None):
        g = batch.g
        labels = (nm_init(g, self.num_nm, self.num_piece) if labels is None else labels)
        embs = self.eval_label(g, labels, environment)
        correctness = self.eval_state(embs, environment, batch)
        reward_total = 0
        for i in range(self.experience_step):
            new_labels, action, _ = self.get_action(mover, labels, g, eps)
            move_node, select_node = action
            embs = self.eval_label(g, new_labels, environment)
            new_correctness = self.eval_state(embs, environment, batch)
            reward = new_correctness - correctness
            reward_total += reward
            experience_data = zip(batch.index.to("cpu"), labels.transpose(0, 1).to("cpu"),
                new_labels.transpose(0, 1).to("cpu"), select_node.transpose(0, 1).to("cpu"),
                move_node.transpose(0, 1).to("cpu"), reward.to("cpu"), )
            experience = [self.Experience._make(dt) for dt in experience_data]
            self.buffer.add(experience)
            labels = new_labels
            correctness = new_correctness
        return reward_total

    def search(self, mover, environment, g, k, eps, labels=None, batch=None):
        """
        Generate a random node marking an search for best node marking for
        k steps.
        """
        labels = nm_init(g, self.num_nm, self.num_piece)
        for j in range(k):
            labels, action, _ = self.get_action(mover, labels, g, eps)
        return labels

    def step(self, mover, memory, *args):
        max_q, _, _ = mover(memory.g, memory.nlabel.transpose(0, 1).contiguous())
        return max_q

    def get_max_action_val(self, mover, memory):
        q_val, context = mover.return_action_q(memory.g, memory.plabel.transpose(0, 1).contiguous(),
            memory.select_node.transpose(0, 1).contiguous(), memory.move_node.transpose(0, 1).contiguous(), )
        return q_val

    def replay(self, model, target, memory):
        q_val = self.get_max_action_val(model, memory)
        with torch.no_grad():
            max_q = self.step(target, memory)
            max_q = max_q.detach()
        return q_val, max_q

    def buffer_reset(self):
        self.buffer.reset()
