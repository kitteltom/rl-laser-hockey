# Code based on https://github.com/sfujim/TD3 and https://github.com/rlcode/per, with major modifications.

import numpy as np
import torch

from sum_tree import SumTree


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def observation_moments(self):
        obs_mean = np.mean(self.state, axis=0)
        obs_std = np.std(self.state, axis=0)
        return obs_mean, obs_std


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_size=int(1e6),
            total_t=int(1e6),
            alpha=0.6,
            beta=0.4,
            beta_schedule="annealing",
            eps=0.01
    ):
        super().__init__(state_dim, action_dim, max_size)

        self.alpha = alpha
        self.beta = beta
        self.delta_beta = (1.0 - beta) / total_t if beta_schedule == "annealing" else 0.0
        self.eps = eps

        self.tree = SumTree(max_size)

    def _get_priority(self, delta):
        # Proportional priority
        return (np.abs(delta) + self.eps) ** self.alpha

    def add(self, state, action, next_state, reward, done):
        self.tree.add(self.tree.max_p, self.ptr)
        super().add(state, action, next_state, reward, done)

    def sample(self, batch_size):
        data_indices = []
        indices = []
        priorities = []

        # Approximate cumulative density with segments of equal probability
        segment = self.tree.total_p() / batch_size

        # Annealing the amount of importance-sampling over time by increasing beta
        self.beta = np.min([1., self.beta + self.delta_beta])

        for i in range(batch_size):
            # Sample exactly one transition from each segment
            lower_bound = segment * i
            upper_bound = segment * (i + 1)
            sample_p = np.random.uniform(lower_bound, upper_bound)

            (idx, p, data_idx) = self.tree.get(sample_p)
            data_indices.append(data_idx)
            indices.append(idx)
            priorities.append(p)

        # Correct sampling bias by using importance-sampling weights
        sampling_probabilities = np.asarray(priorities) / self.tree.total_p()
        importance_weights = np.power(self.size * sampling_probabilities, -self.beta)

        # Normalize weights for stability reasons
        importance_weights /= importance_weights.max()
        importance_weights = torch.FloatTensor(importance_weights).reshape(-1, 1).to(self.device)

        batch = (
            torch.FloatTensor(self.state[data_indices]).to(self.device),
            torch.FloatTensor(self.action[data_indices]).to(self.device),
            torch.FloatTensor(self.next_state[data_indices]).to(self.device),
            torch.FloatTensor(self.reward[data_indices]).to(self.device),
            torch.FloatTensor(self.not_done[data_indices]).to(self.device)
        )

        return batch, indices, importance_weights

    def update(self, idx, delta):
        p = self._get_priority(delta)
        self.tree.update(idx, p)
