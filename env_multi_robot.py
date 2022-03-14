#   Museum environment setup
import numpy as np
from random import choice
class env_multi_robot:
    def __init__(
            self,
            graph,
            trans_prob_matrix,
            stateSpace,
            robotState,
            targetState,
            robotNum,
    ):
        # self.actionVec_ = stateSpace
        # self.stateVec_ = stateSpace
        # self.graph_ = self.graphdict_setup()
        self.reset_robot_state_ = robotState
        self.reset_target_state_ = targetState
        self.robot_num_ = robotNum
        self.graph_ = graph
        self.trans_prob_matrix = trans_prob_matrix
        self.state_space_ = stateSpace
        self.robot_states_ = [robotState for i in range(robotNum)]
        self.target_state_ = targetState
        self.done_ = False
        self.eta_fp_ = 0.0
        self.eta_fn_ = 0.0

    def graphdict_setup(self):
        graph = dict()
        # graph.update({0: [0, 5]})
        # graph.update({1: [1, 2, 5, 7, 9]})
        # graph.update({2: [1, 2, 3, 4]})
        # graph.update({3: [2, 3]})
        # graph.update({4: [2, 4]})
        # graph.update({5: [0, 1, 5]})
        # graph.update({6: [6, 7]})
        # graph.update({7: [1, 6, 7, 8]})
        # graph.update({8: [7, 8]})
        # graph.update({9: [1, 9, 10]})
        # graph.update({10: [9, 10, 11, 12, 13, 18, 19, 20]})
        # graph.update({11: [10, 11, 14]})
        # graph.update({12: [10, 12, 15, 16]})
        # graph.update({13: [10, 13, 17]})
        # graph.update({14: [11, 14, 15]})
        # graph.update({15: [12, 14, 15]})
        # graph.update({16: [12, 16, 17]})
        # graph.update({17: [13, 16, 17]})
        # graph.update({18: [10, 18, 19, 26]})
        # graph.update({19: [10, 18, 19, 20, 25, 26]})
        # graph.update({20: [10, 19, 20, 21, 24]})
        # graph.update({21: [20, 21, 22]})
        # graph.update({22: [21, 22, 23]})
        # graph.update({23: [22, 23, 24]})
        # graph.update({24: [20, 23, 24, 25]})
        # graph.update({25: [19, 24, 25, 26]})
        # graph.update({26: [18, 19, 25, 26]})
        # edges = [(1, 2), (1, 5), (1, 9), (1, 7),
        #          (2, 1), (2, 3), (2, 4),
        #          (3, 2),
        #          (4, 2),
        #          (5, 1),
        #          (6, 7),
        #          (7, 6), (7, 8), (7, 1),
        #          (8, 7),
        #          (9, 1), (9, 10),
        #          (10, 9), (10, 11), (10, 12), (10, 13), (10, 18), (10, 19), (10, 20), (10, 27),
        #          (11, 10), (11, 14), (11, 28),
        #          (12, 10), (12, 15), (12, 16),
        #          (13, 10), (13, 17),
        #          (14, 11), (14, 15),
        #          (15, 14), (15, 12),
        #          (16, 12), (16, 17),
        #          (17, 16), (17, 13),
        #          (18, 10), (18, 19), (18, 26), (18, 70),
        #          (19, 10), (19, 18), (19, 20), (19, 25), (19, 26),
        #          (20, 10), (20, 19), (20, 21), (20, 24),
        #          (21, 20), (21, 22),
        #          (22, 21), (22, 23),
        #          (23, 22), (23, 24),
        #          (24, 23), (24, 20), (24, 25),
        #          (25, 24), (25, 19), (25, 26),
        #          (26, 25), (26, 19), (26, 18),
        #          (27, 10), (27, 49),
        #          (28, 11), (28, 29),
        #          (29, 28), (29, 34),
        #          (30, 31), (30, 33),
        #          (31, 30), (31, 32),
        #          (32, 33), (32, 31), (32, 37),
        #          (33, 32), (33, 36),
        #          (34, 29), (34, 35),
        #          (35, 34), (35, 36), (35, 40), (35, 49),
        #          (36, 35), (36, 33), (36, 39), (36, 37),
        #          (37, 36), (37, 32), (37, 38),
        #          (38, 37), (38, 42),
        #          (39, 36), (39, 40), (39, 41),
        #          (40, 35), (40, 39), (40, 43),
        #          (41, 39), (41, 42),
        #          (42, 41), (42, 38),
        #          (43, 40), (43, 44), (43, 47),
        #          (44, 43), (44, 45),
        #          (45, 44), (45, 46),
        #          (46, 45), (46, 47),
        #          (47, 43), (47, 46), (47, 48),
        #          (48, 47), (48, 50),
        #          (49, 35), (49, 27), (49, 58), (49, 50),
        #          (50, 48), (50, 49), (50, 51), (50, 52),
        #          (51, 50),
        #          (52, 50), (52, 56), (52, 53),
        #          (53, 52), (53, 54), (53, 55), (53, 56),
        #          (54, 53), (54, 55),
        #          (55, 54), (55, 53),
        #          (56, 52), (56, 53), (56, 64),
        #          (57, 58), (57, 63),
        #          (58, 49), (58, 57), (58, 59), (58, 62),
        #          (59, 60), (59, 61), (59, 58),
        #          (60, 70), (60, 61), (60, 59),
        #          (61, 60), (61, 59), (61, 69), (61, 68),
        #          (62, 58), (62, 67),
        #          (63, 57), (63, 66), (63, 64),
        #          (64, 63), (64, 56), (64, 65),
        #          (65, 64), (65, 66),
        #          (66, 65), (66, 67), (66, 63),
        #          (67, 62), (67, 66), (67, 68),
        #          (68, 61), (68, 67), (68, 69),
        #          (69, 61), (69, 68),
        #          (70, 18), (70, 60)]
        edges = [(0, 55), (1, 55), (2, 55), (3, 55),
                     (55, 56), (55, 54), (55, 34), (55, 33), (55, 35),
                     (54, 32), (54, 31), (54, 30), (54, 53),
                     (53, 29), (53, 28), (53, 27), (53, 26), (53, 39), (53, 52), (53, 38), (53, 37), (53, 36),
                     (52, 57), (52, 25), (52, 50),
                     (50, 51), (50, 49),
                     (25, 24), (24, 43),
                     (57, 56), (57, 4), (57, 5), (57, 6), (57, 40), (57, 41), (57, 43),
                     (49, 48), (48, 43), (48, 59),
                     (43, 42), (43, 44), (43, 22),
                     (44, 7), (44, 8), (44, 9), (44, 21), (44, 23), (44, 45),
                     (21, 45), (21, 22),
                     (46, 45), (45, 10), (46, 11), (46, 12), (46, 13), (46, 47), (46, 20), (47, 14),
                     (58, 47), (58, 15), (58, 59), (58, 18), (58, 19), (59, 17), (59, 16)]
        for i in range(71):
            graph.update({i: [i]})
        for edge in edges:
            graph[edge[0]].append(edge[1])
        graph[0].append(5)
        return graph

    def hmm_simulator(self):
        hmm_state = np.zeros(self.state_space_, dtype=float, order='C')
        for i in range(self.state_space_):
            if i == self.target_state_:
                hmm_state[i] = 0.5
                # hmm_state[i] = 1.0
            else:
                hmm_state[i] = 0.5 / (self.state_space_ - 1)
                # hmm_state[i] = 0.
        return hmm_state

    # def checkRobotAction(self, action):
    def update_env(self, action):
        # 更新robot位置
        rewards = np.zeros(self.robot_num_, dtype=int, order='C')
        for i in range(self.robot_num_):
            if action[i] in self.graph_[self.robot_states_[i]]:
                self.robot_states_[i] = action[i]
                # if self.robot_states_[i] == self.target_state_:
                #     self.done_ = True
                #     rewards[i] = -1     #100
                # else:
                #     rewards[i] = -1
                rewards[i] = -1
            else:
                rewards[i] = -2
            # update target pose
            next = np.random.choice(range(len(self.trans_prob_matrix[self.target_state_])), p=self.trans_prob_matrix[self.target_state_])
            if next != 0:
                self.target_state_ = next
        return rewards
        # self.targetState_ = choice(self.graph_[self.targetState_])

    def generate_robot_obs(self):
        obs_robot = []
        for i in range(self.robot_num_):
            cur = np.zeros(self.state_space_, dtype=float, order='C')
            cur[self.robot_states_[i]] = 1.0
            obs_robot.append(cur)
        # obs_target = self.hmm_simulator()
        # obs_robot = np.hstack((obs_robot, obs_target))
        return obs_robot

    def update_env_obs(self):
        obs_env = []
        for robot_state in self.robot_states_:
            if robot_state == self.target_state_:
                a = np.random.rand()
                if a < self.eta_fn_:
                    obs_env.append(0)
                else:
                    obs_env.append(1)
            else:
                a = np.random.rand()
                if a < self.eta_fp_:
                    obs_env.append(1)
                else:
                    obs_env.append(0)
        return obs_env
    # def get_reward(self):
    #     if self.robotState_ == self.targetState_:
    #         self.done_ = True
    #         reward = 100
    #     else:
    #         reward = -1
    #     return reward

    def step(self, action):
        reward = self.update_env(action)
        # reward = self.get_reward()
        obs_robot = self.generate_robot_obs()
        obs_env = self.update_env_obs()
        for i in range(len(obs_env)):
            if obs_env[i] == 1 and self.robot_states_[i] == self.target_state_:
                self.done_ = True
        # observation[self.targetState_] = 1.0
        true_hmm = self.hmm_simulator()
        return obs_robot, obs_env, reward, self.target_state_, self.done_

    def reset(self):
        self.done_ = False
        # self.graph_ = self.graphdict_setup()
        self.robot_states_ = [self.reset_robot_state_ for i in range(self.robot_num_)]
        self.target_state_ = self.reset_target_state_
        true_hmm = self.hmm_simulator()
        obs_robot = self.generate_robot_obs()
        obs_env = self.update_env_obs()
        # observation = self.robotState_
        return obs_robot, obs_env, self.target_state_


