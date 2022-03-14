import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import random
from env_multi_robot import env_multi_robot
from PTB_update import PTB
import matplotlib.pyplot as plt
STATE_SPACE = 61    #71 24 61

def graphdict_setup():
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
    #          (10, 9), (10, 11), (10, 12), (10, 13), (10, 18), (10, 19), (10, 20),
    #          (11, 10), (11, 14),
    #          (12, 10), (12, 15), (12, 16),
    #          (13, 10), (13, 17),
    #          (14, 11), (14, 15),
    #          (15, 14), (15, 12),
    #          (16, 12), (16, 17),
    #          (17, 16), (17, 13),
    #          (18, 10), (18, 19), (18, 23),
    #          (19, 10), (19, 18), (19, 20), (19, 23), (19, 22),
    #          (20, 10), (20, 19), (20, 21),
    #          (21, 20), (21, 22),
    #          (22, 21), (22, 23), (22, 19),
    #          (23, 22), (23, 18), (23, 19)]

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
    #office
    # edges = [(0, 55), (1, 55), (2, 55), (3, 55),
    #                  (55, 56), (55, 54), (55, 34), (55, 33), (55, 35),
    #                  (54, 32), (54, 31), (54, 30), (54, 53),
    #                  (53, 29), (53, 28), (53, 27), (53, 26), (53, 39), (53, 52), (53, 38), (53, 37), (53, 36),
    #                  (52, 57), (52, 25), (52, 50),
    #                  (50, 51), (50, 49),
    #                  (25, 24), (24, 43),
    #                  (57, 56), (57, 4), (57, 5), (57, 6), (57, 40), (57, 41), (57, 43),
    #                  (49, 48), (48, 43), (48, 59),
    #                  (43, 42), (43, 44), (43, 22),
    #                  (44, 7), (44, 8), (44, 9), (44, 21), (44, 23), (44, 45),
    #                  (21, 45), (21, 22),
    #                  (46, 45), (45, 10), (46, 11), (46, 12), (46, 13), (46, 47), (46, 20), (47, 14),
    #                  (58, 47), (58, 15), (58, 59), (58, 18), (58, 19), (59, 17), (59, 16)]
    edges = [(1, 56),
             (2, 56),
             (3, 56),
             (4, 56),
             (5, 58),
             (6, 58),
             (7, 58),
             (8, 45),
             (9, 45),
             (10, 45),
             (11, 46),
             (12, 47),
             (13, 47),
             (14, 47),
             (15, 48),
             (16, 59),
             (17, 60),
             (18, 60),
             (19, 59),
             (20, 59),
             (21, 47),
             (22, 46), (22, 23), (22, 45),
             (23, 22), (23, 44),
             (24, 45),
             (25, 44), (25, 26), (25, 50),
             (26, 25), (26, 53),
             (27, 54),
             (28, 54),
             (29, 54),
             (30, 54),
             (34, 55),
             (32, 55),
             (33, 55),
             (34, 56),
             (35, 56),
             (36, 56),
             (37, 54),
             (38, 54),
             (39, 54),
             (40, 54),
             (41, 58),
             (42, 58),
             (43, 44),
             (44, 43), (44, 58), (44, 45), (44, 23), (44, 25),
             (45, 8), (45, 9), (45, 10), (45, 22), (45, 24), (45, 46), (45, 44),
             (46, 11), (46, 47), (46, 45), (46, 22),
             (47, 46), (47, 12), (47, 13), (47, 14), (47, 48), (47, 21),
             (48, 15), (48, 47), (48, 59),
             (49, 44), (49, 60), (49, 50),
             (50, 49), (50, 25), (50, 51),
             (51, 52), (51, 50), (51, 53),
             (52, 51),
             (53, 58), (53, 26), (53, 51), (53, 54),
             (54, 30), (54, 29), (54, 28), (54, 27), (54, 40), (54, 53), (54, 39), (54, 38), (54, 37), (54, 55),
             (55, 33), (55, 32), (55, 31), (55, 54), (55, 56),
             (56, 57), (56, 55), (56, 35), (56, 34), (56, 36), (55, 1), (55, 2), (55, 3), (55, 4),
             (57, 56), (57, 58),
             (58, 57), (58, 5), (58, 6), (58, 7), (58, 41), (58, 42), (58, 44), (58, 53),
             (59, 48), (59, 16), (59, 60), (59, 19), (59, 20),
             (60, 18), (60, 17), (60, 49), (60, 59)]
    for i in range(STATE_SPACE):
        graph.update({i: [i]})
    for edge in edges:
        graph[edge[0]].append(edge[1])
    graph[0].append(1)
    return graph

def manualInitEmm(graph):
    emm = np.zeros((STATE_SPACE, STATE_SPACE), dtype=float)
    for i in range(len(emm)):
        size = len(graph[i])
        if size == 1:
            emm[i][i] = 1.
            continue
        for j in range(len(emm)):
            if i == j:
                emm[i][j] = 0.9
            elif j in graph[i]:
                emm[i][j] = 0.1 / (size - 1)
            else:
                emm[i][j] = 0.
    return emm







class Policy(nn.Module):
    ## softmax policy
    def __init__(self):
        super(Policy,self).__init__()
        self.affline1 = nn.Linear(STATE_SPACE * 2, STATE_SPACE * 4)     #128
        self.dropout = nn.Dropout(p=0.6)
        self.affline2 = nn.Linear(STATE_SPACE * 4, STATE_SPACE)  #

        self.saved_log_probs = []
        self.saved_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affline1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affline2(x)
        return F.softmax(action_scores,dim=1)


ROBOT_NUM = 5
gamma = 0.01
policy = [Policy() for i in range(ROBOT_NUM)]
# for i in range(ROBOT_NUM):
#     name = '\model' + str(0) + '.pkl'
#     policy[i] = torch.load(name)
optimizer = []
for i in range(ROBOT_NUM):
    # policy[i].cuda()
    optimizer.append(optim.Adam(policy[i].parameters(), lr=0.001))
beta = [0.9 for i in range(ROBOT_NUM)]
eps = 0.01
policy_collector = [[] for i in range(ROBOT_NUM)]

def get_robot_state(observation):
    for i in range(len(observation)):
        if observation[i] == 1.:
            return i
def select_action(state, num):
    robot_state = get_robot_state(state)
    state = torch.from_numpy(state).float().unsqueeze(0)
    #print(state.shape)   torch.size([1,4])
    probs = policy[num](state)
    prob_weights = probs.clone()
    prob_sum = 0.
    for i in graph_[robot_state]:
        prob_sum += probs[0][i]
    # if prob_sum.isnan():
    #     print(prob_sum)
    if prob_sum == 0.:
        length = len(graph_[robot_state])
        for i in range(len(probs[0])):
            if i in graph_[robot_state]:
                prob_weights[0][i] = 1.0 / length
            else:
                prob_weights[0][i] = 0.
    else:
        for i in range(len(probs[0])):
            if i in graph_[robot_state]:
                prob_weights[0][i] = probs[0][i] / prob_sum
            else:
                prob_weights[0][i] = 0.
    m = Categorical(prob_weights)      # distribution
    action = m.sample()           #  sample
    policy[num].saved_log_probs.append(m.log_prob(action))    # logπ(s,a)
    policy[num].saved_probs.append(prob_weights[0][action])
    return action.item()

def finish_episode(num):
    R = 0
    pg_loss = []
    ce_loss = []
    returns = []
    for r in policy[num].rewards[::-1]:
        R = r + gamma * R
        returns.insert(0,R)
    returns = torch.tensor(returns)
    if len(returns) <= 1:
        std = 0.
    else:
        std = returns.std()
    returns = (returns - returns.mean()) / (std + eps)
    for log_prob, R in zip(policy[num].saved_log_probs, returns):
        pg_loss.append(-log_prob * R)          # cross_entrophy
    for j in range(ROBOT_NUM):
        T = len(policy[j].saved_probs)
        if j != num:
            cur_loss = []
            for log_prob_i, prob_j in zip(policy[num].saved_log_probs, policy[j].saved_probs):
                cur_loss.append(log_prob_i * prob_j)
            ce_loss.append(sum(cur_loss) / T)
    optimizer[num].zero_grad()
    a = torch.cat(pg_loss).sum() * beta[num]
    b = torch.cat(ce_loss).sum() * ((1 - beta[num]) / (ROBOT_NUM - 1))
    loss = a + b
    # loss = torch.cat(pg_loss).sum() * beta[num] + torch.cat(ce_loss).sum() * ((1 - beta[num]) / (ROBOT_NUM - 1))         #
    if num == ROBOT_NUM - 1:
        loss.backward()
    else:
        loss.backward(retain_graph=True)
    optimizer[num].step()
    del policy[num].rewards[:]          #
    del policy[num].saved_log_probs[:]



graph_ = graphdict_setup()
transition_prob_ = manualInitEmm(graph_)
fp_prob_ = 0.0
fn_prob_ = 0.0
start_prob_ = np.zeros(STATE_SPACE, dtype=float)
for i in range(len(start_prob_)):
    start_prob_[i] = 1.0 / STATE_SPACE


def main():
    running_reward = 10
    ReturnsCollector = []
    limit = 1000
    for i_episode in range(60000):
        ep_reward = 0
        # initRobotState = random.randint(0, 26)
        initRobotState = 43      #43
        initTargetState = random.randint(1, STATE_SPACE - 1)
        print(initRobotState, initTargetState)
        if i_episode == 20000:
            limit = 30
        env = env_multi_robot(graph_, transition_prob_, STATE_SPACE, initRobotState, initTargetState, ROBOT_NUM)
        one_hot_pose, obs, trueTargetState = env.reset()  #
        estimator = [PTB(graph_, start_prob_, transition_prob_, fp_prob_, fn_prob_) for i in range(ROBOT_NUM)]
        states = [[] for i in range(ROBOT_NUM)]
        for i in range(ROBOT_NUM):
            states[i] = np.hstack((one_hot_pose[i], start_prob_))
            # states[i] = np.hstack((one_hot_pose[i], trueTargetState))
        for t in range(1, 1000):
            actions = [[] for i in range(ROBOT_NUM)]
            robot_poses = [[] for i in range(ROBOT_NUM)]
            for num in range(ROBOT_NUM):
                actions[num] = select_action(states[num], num)
                robot_poses[num] = get_robot_state(one_hot_pose[num])
            one_hot_pose, obs, rewards, trueTargetState, done = env.step(actions)
            for num in range(ROBOT_NUM):
                hmm_ptb = estimator[num].update_ptb(robot_poses[num], obs)
                states[num] = np.hstack((one_hot_pose[num], hmm_ptb))
                # states[num] = np.hstack((one_hot_pose[num], trueTargetState))
                policy[num].rewards.append(rewards[num])
            # ep_reward += reward
            if done:
                print('step = ', t)
                print('i_episode = ', i_episode)
                break
        # running_reward = 0.05 * ep_reward + (1-0.05) * running_reward
        for num in range(ROBOT_NUM):
            finish_episode(num)
        for num in range(ROBOT_NUM):
            del policy[num].saved_probs[:]
            # del policy[num].rewards[:]
            del policy[num].saved_log_probs[:]
            # print(policy[num].saved_probs)

        if i_episode % 500 == 0:
            steps = 0
            times = 100
            for i in range(times):
                initRobotState = 1
                initTargetState = random.randint(1, STATE_SPACE - 1)
                print(initRobotState, initTargetState)
                # start_prob_ = np.zeros(STATE_SPACE, dtype=float)
                # lenTargetState = len(graph_[initTargetState])
                # for t in range(len(start_prob_)):
                #     if t in graph_[initTargetState]:
                #         start_prob_[t] = 0.5 / lenTargetState
                #     else:
                #         start_prob_[t] = 0.5 / (STATE_SPACE - lenTargetState)
                env = env_multi_robot(graph_, transition_prob_, STATE_SPACE, initRobotState, initTargetState, ROBOT_NUM)
                one_hot_pose, obs, trueTargetState = env.reset()  # ep_reward表示每个episode中的reward
                estimator = [PTB(graph_, start_prob_, transition_prob_, fp_prob_, fn_prob_) for i in range(ROBOT_NUM)]
                states = [[] for i in range(ROBOT_NUM)]
                for i in range(ROBOT_NUM):
                    states[i] = np.hstack((one_hot_pose[i], start_prob_))
                    # states[i] = np.hstack((one_hot_pose[i], trueTargetState))
                for t in range(1, limit):
                    actions = [[] for i in range(ROBOT_NUM)]
                    robot_poses = [[] for i in range(ROBOT_NUM)]
                    for num in range(ROBOT_NUM):
                        actions[num] = select_action(states[num], num)
                        robot_poses[num] = get_robot_state(one_hot_pose[num])
                    one_hot_pose, obs, rewards, trueTargetState, done = env.step(actions)
                    for num in range(ROBOT_NUM):
                        hmm_ptb = estimator[num].update_ptb(robot_poses[num], obs)
                        states[num] = np.hstack((one_hot_pose[num], hmm_ptb))
                        # states[num] = np.hstack((one_hot_pose[num], trueTargetState))
                    steps += 1
                    if done:
                        print('step = ', t)
                        break
                # running_reward = 0.05 * ep_reward + (1-0.05) * running_reward
                # for num in range(ROBOT_NUM):
                #     finish_episode(num)
                for num in range(ROBOT_NUM):
                    del policy[num].saved_probs[:]
                    # del policy[num].rewards[:]
                    del policy[num].saved_log_probs[:]
            average_step = steps / times
            ReturnsCollector.append(average_step)
            print(ReturnsCollector)
            # plt.plot(ReturnsCollector)
            # plt.xlabel('episode steps / 10')
            # plt.ylabel('average test returns')
            # plt.show()
            # torch.save(net.state_dict(), PATH)
    for i in range(ROBOT_NUM):
        name = '\model' + str(i) + '.pkl'
        torch.save(policy, name)
    print(ReturnsCollector)

if __name__ == '__main__':
    main()