# Code based on https://github.com/rlcode/per.

import numpy as np


class SumTree:
    # A binary tree data structure where the parent’s value is the sum of its children
    def __init__(self, max_size):
        self.max_size = max_size
        self.max_p = 1.

        self.tree = np.zeros(2 * max_size - 1)

    # Recursively propagate the update to the root node
    def _propagate(self, idx, delta_p):
        parent = (idx - 1) // 2

        self.tree[parent] += delta_p

        if parent != 0:
            self._propagate(parent, delta_p)

    # Find sample on leaf node
    def _retrieve(self, idx, sample_p):
        left = 2 * idx + 1
        right = left + 1

        if left >= self.tree.size:
            return idx

        if sample_p <= self.tree[left]:
            return self._retrieve(left, sample_p)
        else:
            return self._retrieve(right, sample_p - self.tree[left])

    # Return total priority
    def total_p(self):
        return self.tree[0]

    # Store priority
    def add(self, p, ptr):
        idx = ptr + self.max_size - 1
        self.update(idx, p)

    # Update priority
    def update(self, idx, p):
        delta_p = p - self.tree[idx]

        self.tree[idx] = p
        if p > self.max_p:
            print(f' -> Current maximal priority in the replay buffer: {p:.3f}')
        self.max_p = max(p, self.max_p)
        self._propagate(idx, delta_p)

    # Get priority and data index
    def get(self, sample_p):
        idx = self._retrieve(0, sample_p)
        data_idx = idx - self.max_size + 1

        return idx, self.tree[idx], data_idx
