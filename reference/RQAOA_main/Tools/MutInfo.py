import numpy as np
from itertools import product
from scipy.linalg import logm
import seaborn as sns
import matplotlib.pyplot as plt




class MutualInformationMap:

    def __init__(self, state):

        self.state = state
        self.n_qubits = int(np.log2(len(state)))
        self.den_matrix = np.outer(state.conj(), state)
        self.state_list = 2 * ((np.arange(2 ** self.n_qubits)[:, None]
                           >> np.arange(self.n_qubits - 1, -1, -1)) & 1) - 1


    def _one_q_trace(self, q_ind_1):

        part_trace = [np.sum(self.den_matrix[
                                 (self.state_list[:, q_ind_1] == i),
                                 (self.state_list[:, q_ind_1] == k)])
                      for i, k in product([-1, 1], repeat=2)]

        return np.array(part_trace, dtype='complex128').reshape(2, 2)

    def _two_q_trace(self, q_ind_1, q_ind_2):

        part_trace = [np.sum(self.den_matrix[
                                 (self.state_list[:, q_ind_1] == i) &
                                 (self.state_list[:, q_ind_2] == k),
                                 (self.state_list[:, q_ind_1] == m) &
                                 (self.state_list[:, q_ind_2] == l)])
                      for i, k, m, l in product([-1, 1], repeat=4)]

        return np.array(part_trace, dtype='complex128').reshape(4, 4)

    def mutual_information_map(self):
        MI = np.zeros((self.n_qubits, self.n_qubits), dtype='complex128')
        for i , k in product(range(self.n_qubits), repeat=2):
            if i == k :
                continue
            else:
                S_i = - np.trace(self._one_q_trace(i) @ logm(self._one_q_trace(i)))
                S_k = - np.trace(self._one_q_trace(k) @ logm(self._one_q_trace(k)))
                S_ik = - np.trace(self._two_q_trace(i, k) @ logm(self._two_q_trace(i, k)))
                MI[i, k] = 0.5 * (S_i + S_k - S_ik)

        return np.real(MI)




real_part = np.random.random_sample(1 << 8)
imag_part = np.random.random_sample(1 << 8)

state = real_part + 1j * imag_part
state = (state / np.sqrt(np.sum(np.abs(state) ** 2))).astype(np.complex128)


mim = MutualInformationMap(state=state)

a = mim._one_q_trace(3),
b = mim._one_q_trace(4),
c = mim._two_q_trace(3, 4)

d = np.kron(a, b)


print(np.round(a, 4))
print(np.round(b, 4))

print(np.round(c, 4))

print(np.round(d, 4))



print(np.round(np.abs((c - d)) / np.abs(c), 4))














