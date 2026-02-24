"""
Mission of this file:
This file implements the Value Iteration algorithm to solve a Markov Decision Process (MDP) in a Grid World environment. 
Value Iteration computes the optimal state value function by iteratively applying the Bellman Optimality Equation.
Once the state values converge, an optimal policy is directly derived using a greedy approach over the action values.
"""
# --- Standard Library Imports ---
import os      # Operating system interfaces (setting env vars)
import time    # Time access and conversions
import random  # Generate pseudo-random numbers
import sys     # System-specific parameters and functions
from pathlib import Path # Object-oriented filesystem paths

# Suppress TensorFlow oneDNN optimization warnings globally
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# --- Third-Party Library Imports ---
import numpy as np                # Fundamental package for scientific computing
import matplotlib.pyplot as plt   # State-based interface for plotting
from torch.utils import data      # Dataset / DataLoader helpers (currently unused if purely for reinforcement learning)
from torch.utils.tensorboard import SummaryWriter  # TensorBoard logging for metrics

# --- Local Module Imports ---
# Reference parent directory - use absolute path for cross-platform compatibility to import Utils module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Utils"))
import grid_env   # Local grid environment definition for MDP experiments


class class_value_iteration():
    """
    Mission of the class:
    To encapsulate the properties and methods required for running the Value Iteration algorithm on a given Grid World environment.
    It manages the state values, Q-values, and the current policy.
    """
    
    def __init__(self, env: grid_env.GridEnv):
        """
        Mission of the code-block:
        Initialize the Value Iteration solver with the environment and preset algorithmic parameters.
        
        Logic: 
        Extracts needed space sizes from the passed environment, initializes the state values and Q-table 
        to zeros, and sets up a uniform initial policy. It also initializes TensorBoard logging.
        
        Inputs:
        - env (grid_env.GridEnv): The grid world environment instance.
        
        Outputs:
        - None
        """
        self.gama = 0.9   #discount rate
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size**2  # Exponentiation, grid world size e.g. 5 ** 2 = 25 grid world.
        self.reward_space_size, self.reward_list = len(self.env.reward_list), self.env.reward_list  # In parent class: self.reward_list = [0, 1, -10, -10]
        #state_value
        self.state_value = np.zeros(shape=self.state_space_size)  # 1D array
        #action value -> Q-table
        self.qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size)) # 25 x 5

        self.mean_policy = np.ones(shape=(self.state_space_size, self.action_space_size)) / self.action_space_size
        self.policy = self.mean_policy.copy()
        self.writer = SummaryWriter("../logs")  # Instantiate SummaryWriter object

        print("action_space_size: {} state_space_size：{}" .format(self.action_space_size ,self.state_space_size) )
        print("state_value.shape:{} , qvalue.shape:{} , mean_policy.shape:{}".format(self.state_value.shape,self.qvalue.shape, self.mean_policy.shape))
        print("\nRespectively non-forbidden area, target area, forbidden area and hitting wall:")
        print("self.reward_space_size:{},self.reward_list:{}".format(self.reward_space_size,self.reward_list))
        print('----------------------------------------------------------------')

    def value_iteration_new(self, tolerance=0.001, steps=100):
        """
        Mission of the code-block:
        Iteratively solve optimal Bellman equation to get optimal state value and derive the policy.
        
        Logic:
        Continuously updates `state_value` using maximum Q-values until the L1 norm between consecutive 
        state estimations is below the `tolerance` threshold, or max `steps` is reached. Inside the loop, 
        it performs policy improvements by acting greedily with respect to the updated Q-values.
        
        Inputs:
        - tolerance (float): When the norm of state_value before and after is less than tolerance, state_value is considered converged
        - steps (int): Stop when iteration count exceeds steps. Recommend setting this variable larger
        
        Outputs:
        - steps (int): Remaining iteration count
        """
        # Initialize V0 to 1
        state_value_k = np.ones(self.state_space_size)
        while np.linalg.norm(state_value_k - self.state_value, ord=1)>tolerance and steps>0:
            steps -= 1
            self.state_value = state_value_k.copy()
            """
                  A variant of normal policy_improvement, equivalent to the value iteration algorithm. Also usable for policy iteration. When doing policy iteration, the second return value does not need to be received
                  Update qvalue ; qvalue[state,action]=reward+value[next_state]
                  Find action* at state: action* = arg max(qvalue[state,action]), i.e., the optimal action corresponds to the maximum qvalue
                  Update policy: Set probability of action* to 1 and other actions to 0. This is a greedy policy
                  :param: state_value: State value corresponding to policy
                  :return: Improved policy, and state_value of next iteration step
            """
            # Method initialized a new policy with probabilities of all actions in all states set to 0
            policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
            #state_value_k = state_value_k.copy()
            # Iterate through all states
            q_table = np.zeros(shape=(self.state_space_size, self.action_space_size))
            for state in range(self.state_space_size):
                qvalue_list = []
                # Iterate through all actions
                for action in range(self.action_space_size):
                    # Calculate qvalue, i.e., action value.
                    """
                    Mission of the inner code-block:
                    Calculate Q-value in elementwise form for the current state-action pair.
                    
                    Logic:
                    Averages the immediate rewards across possible transitions and adds the discounted 
                    expected future return landing on all next possible states.
                    
                    Inputs: 
                    - state (int): Corresponding state index from outer loop
                    - action (int): Corresponding action index from inner loop
                    
                    Outputs: 
                    Calculates local `qvalue` which is appended to `qvalue_list`.
                    """
                    qvalue = 0
                    for i in range(self.reward_space_size):
                        # print("self.reward_list[i] * self.env.Rsa[state, action, i]:{}x{}={}".format(self.reward_list[i], self.env.Rsa[state, action, i],self.reward_list[i] * self.env.Rsa[state, action, i]))
                        qvalue += self.reward_list[i] * self.env.Rsa[state, action, i]

                    for next_state in range(self.state_space_size):
                        qvalue += self.gama * self.env.Psa[state, action, next_state] * state_value_k[next_state]
                    qvalue_list.append(qvalue)
                # print("qvalue_list:",qvalue_list)
                q_table[state,:] = qvalue_list.copy()

                state_value_k[state] = max(qvalue_list)  # Get max state value of this state
                action_star = qvalue_list.index(max(qvalue_list))  # Get action corresponding to max state value
                policy[state, action_star] = 1  # Update policy, greedy algorithm
            print("q_table:{}".format(q_table))
            self.policy = policy
            
            # --- Visualize policy per iteration ---
            if self.env.render_.ax is not None:
                self.env.render_.ax.cla()
            self.env.render_._inited = False
            self.env.render_._ensure_axes()

            self.env.render_.trajectory.clear() # clear past renders if needed
            self.show_policy()
            self.show_state_value(self.state_value, y_offset=0.25)
            plt.pause(0.1)
            self.env.render(block=False) # Interactive non-blocking render
            
        return steps

    def show_policy(self):
        """
        Mission of the code-block:
        Visualize the currently learned policy on the grid world environment render.
        
        Logic:
        Iterates over all states and actions, drawing action arrows proportional to the probability 
        of taking each action in each state on the environment's UI.
        
        Inputs: None
        Outputs: None (Updates the environment UI visualization)
        """
        for state in range(self.state_space_size):
            for action in range(self.action_space_size):
                policy = self.policy[state, action]
                self.env.render_.draw_action(pos=self.env.state2pos(state),
                                             toward=policy * 0.4 * self.env.action_to_direction[action],
                                             radius=policy * 0.1)

    def show_state_value(self, state_value, y_offset=0.2):
        """
        Mission of the code-block:
        Visualize the computed state values on the grid world environment render.
        
        Logic:
        Iterates over all states and writes the rounded state value text on the corresponding cell in the UI.
        
        Inputs:
        - state_value (np.ndarray): The array of state values to print.
        - y_offset (float): Offset for placing text vertically in the grid cells.
        
        Outputs: None (Updates the environment UI visualization)
        """
        for state in range(self.state_space_size):
            self.env.render_.write_word(pos=self.env.state2pos(state), word=str(round(state_value[state], 1)),
                                        y_offset=y_offset,
                                        size_discount=0.7)


    def obtain_episode(self, policy, start_state, start_action, length):
        """
        Mission of the code-block:
        Simulate an episode following a designated policy starting from a specific state and action.
        
        Logic:
        Simulates `length` transitions in the environment using the provided policy.
        Records every state, action, reward, and the resulting state/action into an episode list.
        
        Inputs:
        - policy (np.ndarray): Generate episode from designated policy (action probabilities matrix)
        - start_state (int): Starting state ID
        - start_action (int): Starting action ID
        - length (int): Episode length
        
        Outputs:
        - episode (list of dicts): A sequence of state, action, reward, next_state, next_action dicts
        """
        self.env.agent_location = self.env.state2pos(start_state)
        episode = []
        next_action = start_action
        next_state = start_state
        while length > 0:
            length -= 1
            state = next_state
            action = next_action
            _, reward, done, _, _ = self.env.step(action)
            next_state = self.env.pos2state(self.env.agent_location)
            next_action = np.random.choice(np.arange(len(policy[next_state])),
                                           p=policy[next_state])
            episode.append({"state": state, "action": action, "reward": reward, "next_state": next_state,
                            "next_action": next_action})
        return episode



if __name__ == "__main__":
    """
    Mission of this block:
    Test script execution block to initialize the grid environment, run the Value Iteration solver, 
    and visualize the results.
    
    Logic:
    Creates a 3x3 GridWorld with a specific target and forbidden cells. Instantiates the solver, 
    times the value_iteration_new method, checks if it converged, and then renders the resulting 
    policy and state values in a blocking UI window.
    """
    print("-----Begin!-----")
    gird_world2x2 = grid_env.GridEnv(size=7, target=[6, 6],
                           forbidden=[[1, 0],[2,1],[3,3],[4,1],[0,2],[2,3],[5,5],[3,2],[4,4],[3,0]],
                           render_mode='')

    solver = class_value_iteration(gird_world2x2)
    start_time = time.time()

    # Execute value iteration algorithm
    demand_step = 1000
    remaining_steps = solver.value_iteration_new(tolerance=0.1, steps=demand_step)
    if remaining_steps > 0:
        print("Value iteration converged in {} steps.".format(demand_step - remaining_steps))
    else:
        print("Value iteration did not converge in 100 steps.")

    end_time = time.time()

    cost_time = end_time - start_time
    print("cost_time:{}".format(round(cost_time, 2)))
    print(len(gird_world2x2.render_.trajectory))

    solver.show_policy()  # solver.env.render()
    solver.show_state_value(solver.state_value, y_offset=0.25)

    print("Close the window to end the program.")
    gird_world2x2.render(block=True)