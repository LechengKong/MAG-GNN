import numpy as np
import time
import torch
from multiprocessing import Array, Manager

from dgl.dataloading.dataloader import GraphCollator

from collections import deque, namedtuple
from gp.utils.datasets import DatasetWithCollate


"""
Following namedtuple makes data collating nad referencing easier
"""

GraphLabelNT = namedtuple("GraphLabelNT", ["g", "labels", "index"])
ReplayBatch = namedtuple(
    "ReplayBatch",
    ["g", "plabel", "nlabel", "select_node", "move_node", "reward"],
)
GraphNMLabel = namedtuple("GraphNMLabel", ["g", "rlabel", "labels", "index"])

ReplayBatchHist = namedtuple(
    "ReplayBatchHist",
    [
        "g",
        "plabel",
        "nlabel",
        "select_node",
        "move_node",
        "reward",
        "context",
    ],
)

ReplayBatchSM = namedtuple(
    "ReplayBatchSM",
    [
        "g",
        "plabel",
        "nlabel",
        "select_node",
        "move_node",
        "reward",
    ],
)


class GraphLabelDataset(DatasetWithCollate):
    def __init__(self, graphs, labels, ind=None) -> None:
        super().__init__()

        self.graphs = graphs
        self.labels = labels
        if ind is None:
            self.ind = np.arange(len(self.graphs))
        else:
            self.ind = ind

    def __getitem__(self, index):
        return GraphLabelNT(
            self.graphs[index], np.array([self.labels[index]]), self.ind[index]
        )

    def __len__(self):
        return len(self.graphs)

    def get_collate_fn(self):
        return GraphCollator().collate


class GraphNMDataset(DatasetWithCollate):
    def __init__(self, graphs, labels, r_labels) -> None:
        super().__init__()

        self.graphs = graphs
        self.labels = labels
        self.rlabels = r_labels

    def __getitem__(self, index):
        return GraphNMLabel(
            self.graphs[index],
            np.array(self.rlabels[index]),
            np.array([self.labels[index]]),
            index,
        )

    def __len__(self):
        return len(self.graphs)

    def get_collate_fn(self):
        return GraphCollator().collate


class GraphReplayBuffer:
    def __init__(self, capacity):
        # self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.buffer = np.empty(capacity, dtype=object)
        self.size_count = 0
        self.pointer = 0

    def __len__(self):
        return self.size_count

    def add(self, experience):
        if isinstance(experience, list):
            ct = len(experience)
            self.size_count += ct
            next_pointer = self.pointer + ct
            if next_pointer > self.capacity:
                reminder = self.capacity - self.pointer
                self.buffer[self.pointer :] = experience[:reminder]
                self.pointer = 0
                next_pointer = next_pointer - self.capacity
                ct = next_pointer
                experience = experience[reminder:]
            self.buffer[self.pointer : next_pointer] = experience
            self.pointer = next_pointer
            self.size_count = min(self.size_count, self.capacity)
        else:
            self.buffer[self.size_count] = experience
        # if (len(self) / self.capacity) > 1.3:
        #     self.buffer = self.buffer[-self.capacity :]

    def sample(self, batch_size):
        if len(self) < batch_size:
            return None
        indices = np.random.choice(len(self), batch_size, replace=True)
        return [self.buffer[idx] for idx in indices]

    def reset(self):
        self.buffer = deque(maxlen=self.capacity)


class ReplayDataset(DatasetWithCollate):
    def __init__(self, graphs, buffer, replay_size=1, replay_type=ReplayBatch):
        super().__init__()
        self.graphs = graphs
        self.buffer = buffer
        self.replay_size = replay_size
        self.replay_type = replay_type

    def __getitem__(self, index):
        # print(len(self.buffer))
        replay = self.buffer.sample(1)[0]
        replay_graph = self.graphs[int(replay[0])]
        return self.replay_type(replay_graph, *map(np.array, replay[1:]))

    def __len__(self):
        return len(self.graphs) * self.replay_size

    def get_collate_fn(self):
        return GraphCollator().collate
