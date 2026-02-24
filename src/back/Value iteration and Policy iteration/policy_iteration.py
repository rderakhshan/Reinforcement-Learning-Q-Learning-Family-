import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import random
import time

import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter

# Reference parent directory - use absolute path for cross-platform compatibility
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Utils"))
import grid_env

class  class_policy_iteration:
    def __init__(self,env: grid_env.GridEnv):
        self.gama = 0.9  # discount rate
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size ** 2  # Exponentiation, grid world size e.g. 5 ** 2 = 25 grid world.
        self.reward_space_size, self.reward_list = len(
            self.env.reward_list), self.env.reward_list  # In parent class: self.reward_list = [0, 1, -10, -10]
        # state_value
        self.state_value = np.zeros(shape=self.state_space_size)  # 1D array
        # action value -> Q-table
        self.qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))  # 25 x 5

        self.mean_policy = np.ones(shape=(self.state_space_size, self.action_space_size)) / self.action_space_size
        self.policy = self.mean_policy.copy()
        self.writer = SummaryWriter("./logs")  # Instantiate SummaryWriter object

        print("action_space_size: {} state_space_size：{}".format(self.action_space_size, self.state_space_size))
        print("state_value.shape:{} , qvalue.shape:{} , mean_p olicy.shape:{}".format(self.state_value.shape,
                                                                                     self.qvalue.shape,
                                                                                     self.mean_policy.shape))
        print("\nRespectively non-forbidden area, target area, forbidden area and hitting wall:")
        print("self.reward_space_size:{},self.reward_list:{}".format(self.reward_space_size, self.reward_list))
        print('----------------------------------------------------------------')

    def random_greed_policy(self):
        """
        Generate a random greedy policy
        :return:
        """
        policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
        for state_index in range(self.state_space_size):
            action = np.random.choice(range(self.action_space_size))  # Randomly select elements from the array
            policy[state_index, action] = 1  # Select this action as the policy, set to 1
        print("random_choice_policy",policy)
        return policy

    def policy_iteration(self,tolerance = 0.001,steps=100):
        """

            :param tolerance: If the norm of the policy before and after iteration is less than tolerance, it is considered converged
            :param steps: When step is small, it degenerates into truncated iteration
            :return: Remaining iteration count
        """
        policy = self.random_greed_policy()
        while np.linalg.norm(policy - self.policy, ord=1) > tolerance and steps > 0:
            steps -= 1
            policy = self.policy.copy()
            self.state_value = self.policy_evaluation(self.policy.copy(), tolerance, steps)
            self.policy, _ = self.policy_improvement(self.state_value)  # Only receive the first return value (more concerned with the first return value)
        return steps

    def policy_evaluation(self, policy, tolerance=0.001, steps=10):
        """
        Iteratively solve Bellman equation to get state value. Satisfying either tolerance or steps is sufficient
        :param policy: Policy to be solved
        :param tolerance: When the norm of state_value before and after is less than tolerance, state_value is considered converged
        :param steps: Stop calculation when iteration count exceeds steps. In policy iteration, the algorithm becomes truncated iteration
        :return: Converged value after solving
        """
        state_value_k = np.ones(self.state_space_size)
        state_value = np.zeros(self.state_space_size)
        while np.linalg.norm(state_value_k - state_value, ord=1) > tolerance:  # While j < jtruncate, do
            state_value = state_value_k.copy()
            for state in range(self.state_space_size):
                value = 0
                for action in range(self.action_space_size):
                    value += policy[state, action] * self.calculate_qvalue(state_value=state_value_k.copy(),
                                                                           state=state,
                                                                           action=action)  # bootstrapping
                state_value_k[state] = value
        return state_value_k

    def calculate_qvalue(self, state, action, state_value):
        """
        Calculate qvalue in elementwise form
        :param state: Corresponding state
        :param action: Corresponding action
        :param state_value: State value
        :return: Calculated result
        """
        qvalue = 0
        for i in range(self.reward_space_size):
            qvalue += self.reward_list[i] * self.env.Rsa[state, action, i]
        for next_state in range(self.state_space_size):
            qvalue += self.gama * self.env.Psa[state, action, next_state] * state_value[next_state]
        return qvalue

    def policy_improvement(self, state_value):
        """
        A variant of normal policy_improvement, equivalent to the value iteration algorithm. Also usable for policy iteration. When doing policy iteration, the second return value does not need to be received
        Update qvalue ; qvalue[state,action]=reward+value[next_state]
        Find action* at state: action* = arg max(qvalue[state,action]), i.e., the optimal action corresponds to the maximum qvalue
        Update policy: Set probability of action* to 1 and other actions to 0. This is a greedy policy
        :param: state_value: State value corresponding to policy
        :return: Improved policy, and state_value of next iteration step
        """
        policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
        state_value_k = state_value.copy()
        for state in range(self.state_space_size):
            qvalue_list = []
            for action in range(self.action_space_size):
                qvalue_list.append(self.calculate_qvalue(state, action, state_value.copy()))
            state_value_k[state] = max(qvalue_list)
            action_star = qvalue_list.index(max(qvalue_list))
            policy[state, action_star] = 1
        return policy, state_value_k



    def show_policy(self):
        for state in range(self.state_space_size):
            for action in range(self.action_space_size):
                policy = self.policy[state, action]
                self.env.render_.draw_action(pos=self.env.state2pos(state),
                                             toward=policy * 0.4 * self.env.action_to_direction[action],
                                             radius=policy * 0.1)

    def show_state_value(self, state_value, y_offset=0.2):
        for state in range(self.state_space_size):
            self.env.render_.write_word(pos=self.env.state2pos(state), word=str(round(state_value[state], 1)),
                                        y_offset=y_offset,
                                        size_discount=0.7)

    def obtain_episode(self, policy, start_state, start_action, length):
        """

        :param policy: Generate episode from designated policy
        :param start_state: Starting state
        :param start_action: Starting action
        :param length: Episode length
        :return: A sequence of state, action, reward, next_state, next_action
        """
        self.env.agent_location = self.env.state2pos(start_state)
        episode = []
        next_action = start_action
        next_state = start_state
        while length > 0:
            length -= 1
            state = next_state
            action = next_action
            _, reward, done, _, _ = (self.env.step+
                                     (action))
            next_state = self.env.pos2state(self.env.agent_location)
            next_action = np.random.choice(np.arange(len(policy[next_state])),
                                           p=policy[next_state])
            episode.append({"state": state, "action": action, "reward": reward, "next_state": next_state,
                            "next_action": next_action})
        return episode


if __name__ == "__main__":
    print("-----Begin!-----")
    gird_world2x2 = grid_env.GridEnv(size=3, target=[2, 2],
                           forbidden=[[1, 0],[2,1]],
                           render_mode='')
    solver = class_policy_iteration(gird_world2x2)
    start_time = time.time()

    demand_step = 10000
    remaining_steps = demand_step - solver.policy_iteration(tolerance=0.001, steps=demand_step)
    if remaining_steps > 0:
        print("Policy iteration converged in {} steps.".format(demand_step - remaining_steps))
    else:
        print("Policy iteration did not converge in {} steps.".format(demand_step))

    end_time = time.time()

    cost_time = end_time - start_time
    print("cost_time:{}".format(round(cost_time, 2)))
    print(len(gird_world2x2.render_.trajectory))

    solver.show_policy()  # solver.env.render()
    solver.show_state_value(solver.state_value, y_offset=0.25)

    gird_world2x2.render(block=True)