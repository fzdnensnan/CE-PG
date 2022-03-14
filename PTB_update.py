import numpy as np

class PTB:
    def __init__(self, graph, start_prob, transititon_prob, fp_prob, fn_prob):
        self.graph_ = graph
        self.state_space_ = len(graph)
        self.ptb_ = start_prob
        self.gamma_ = transititon_prob
        self.eta_fp_ = fp_prob
        self.eta_fn_ = fn_prob
        self.obss_ = []
        self.poss_ = []
        self.lambda_ = np.zeros((self.state_space_, self.state_space_))

    def update_lambda_matrix(self, robot_pos, observation):
        for i in range(len(self.lambda_)):
            if i == robot_pos:
                if observation == 1:
                    self.lambda_[i][i] = 1 - self.eta_fn_
                else:   self.lambda_[i][i] = self.eta_fn_
            else:
                if observation == 1:
                    self.lambda_[i][i] = self.eta_fp_
                else:   self.lambda_[i][i] = 1 - self.eta_fp_

    def store_trajectory(self, robot_pos, observation):
        self.obss_.append(observation)
        self.poss_.append(robot_pos)

    def update_ptb(self, robot_pos, observation):
        self.store_trajectory(robot_pos, observation)
        self.update_lambda_matrix(robot_pos, observation)
        buffer = np.matmul(self.lambda_, self.gamma_)
        self.ptb_ = np.matmul(buffer, self.ptb_)
        sum = 0.
        for i in self.ptb_:
            sum += i
        weight = 1 / sum
        self.ptb_ = self.ptb_ * weight
        return self.ptb_

    def get_current_ptb(self):
        return self.ptb_


def manualInitEmm(graph):
    emm = np.zeros((27, 27), dtype=float)
    for i in range(len(emm)):
        size = len(graph[i])
        if size == 1:
            emm[i][i] = 1.
            continue
        for j in range(len(emm)):
            if i == j:
                emm[i][j] = 0.5
            elif j in graph[i]:
                emm[i][j] = 0.5 / (size - 1)
            else:
                emm[i][j] = 0.
    return emm


graph = dict()
graph.update({0: [0, 5]})
graph.update({1: [1, 2, 5, 7, 9]})
graph.update({2: [1, 2, 3, 4]})
graph.update({3: [2, 3]})
graph.update({4: [2, 4]})
graph.update({5: [0, 1, 5]})
graph.update({6: [6, 7]})
graph.update({7: [1, 6, 7, 8]})
graph.update({8: [7, 8]})
graph.update({9: [1, 9, 10]})
graph.update({10: [9, 10, 11, 12, 13, 18, 19, 20]})
graph.update({11: [10, 11, 14]})
graph.update({12: [10, 12, 15, 16]})
graph.update({13: [10, 13, 17]})
graph.update({14: [11, 14, 15]})
graph.update({15: [12, 14, 15]})
graph.update({16: [12, 16, 17]})
graph.update({17: [13, 16, 17]})
graph.update({18: [10, 18, 19, 26]})
graph.update({19: [10, 18, 19, 20, 25, 26]})
graph.update({20: [10, 19, 20, 21, 24]})
graph.update({21: [20, 21, 22]})
graph.update({22: [21, 22, 23]})
graph.update({23: [22, 23, 24]})
graph.update({24: [20, 23, 24, 25]})
graph.update({25: [19, 24, 25, 26]})
graph.update({26: [18, 19, 25, 26]})

start_prob = np.zeros(27, dtype=float)
for i in range(len(start_prob)):
    start_prob[i] = 1.0 / 27

# start_prob = [[0.5], [0.25], [0.25]]
# transition_prob = [[0.7, 0.3, 0.],
#                    [0.15, 0.7, 0.15],
#                    [0., 0.3, 0.7]]
transition_prob = manualInitEmm(graph)

fp_prob = 0.1
fn_prob = 0.1
estimator = PTB(graph, start_prob, transition_prob, fp_prob, fn_prob)
print('b0:')
print(estimator.ptb_)
robot_pose = 2
obs = 0
res = estimator.update_ptb(robot_pose, obs)
print('b1:')
print(res)



