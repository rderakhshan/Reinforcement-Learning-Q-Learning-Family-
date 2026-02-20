"""
Deep Q-Network Agent core logical interface for TensorFlow.

This module encapsulates the DQN agent methodologies connecting the TensorFlow
Deep Q-Network, Target Network, and underlying Replay Buffer memory logic dynamically.

Typical usage example:
    from agent import Agent
    agent = Agent(gamma=0.99, epsilon=1.0, lr=0.0001, ...)
    action = agent.choose_action(observation)
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import os
from network import DeepQNetwork
from replay_memory import ReplayBuffer


class Agent:
    """Deep Q-Learning Agent managing synchronized neural computations dynamically (TensorFlow).

    Coordinates execution logic traversing the fundamental boundaries linking
    stochastic epsilon-greedy algorithms dynamically natively evaluated across Keras.

    Attributes:
        gamma (float): Discount rate parsing sequential reward parameters dynamically.
        epsilon (float): Dynamic random bounding limiting absolute algorithm convergence.
        lr (float): Agent's localized learning rate scaling structurally.
        n_actions (int): Discrete bounds referencing environment outputs mathematically.
        input_dims (tuple): Bounding dimensions framing structural inputs visually.
        batch_size (int): Total sampled parameter clusters routing memory gradients.
        eps_min (float): Lowest floor evaluated parsing epsilon random boundaries natively.
        eps_dec (float): Iterative step decrement subtracted from epsilon parameters natively.
        replace_target_cnt (int): Iterative ceiling tracking network synchronizations natively.
        algo (str): Identifying string for model saved weights natively.
        env_name (str): Identifying string tracking the environment context natively.
        chkpt_dir (str): Base output directory mapping serialization logic natively.
        action_space (list): Mapped valid integer representations available globally.
        learn_step_counter (int): Current counter evaluating iteration metrics dynamically.
        fname (str): Structuring output prefix globally assigning logical checks safely.
        memory (ReplayBuffer): Retained caching structure containing historical logic.
        q_eval (DeepQNetwork): The actively predicting dynamically tracing network layers natively.
        q_next (DeepQNetwork): The structurally insulated target forecasting layer natively.
    """

    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn'):
        """Initializes Keras agent configurations evaluating external parameters logically.

        Args:
            gamma (float): Numeric discount evaluating successive outcomes flexibly.
            epsilon (float): Random boundary targeting exploration paths structurally.
            lr (float): Optimization mapping constant logically.
            n_actions (int): Output limits explicitly defined structurally.
            input_dims (tuple): Array bounds parsing geometric limits natively.
            mem_size (int): Size constant scaling physical memory limits structurally.
            batch_size (int): Constraints capturing parallel paths safely.
            eps_min (float, optional): Decay boundaries. Defaults to 0.01.
            eps_dec (float, optional): Ratio terminating exploration variables. Defaults to 5e-7.
            replace (int, optional): Frequency updating target weights synchronously. Defaults to 1000.
            algo (str, optional): Algorithm name cleanly mapping directories. Defaults to None.
            env_name (str, optional): Emulator bounds formatting labels logically. Defaults to None.
            chkpt_dir (str, optional): Serialization directory limits intuitively. Defaults to 'tmp/dqn'.
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        
        # Formatting model string names safely combining local structures globally
        self.fname = os.path.join(self.chkpt_dir, self.env_name + '_' + self.algo + '_')

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        # Generating evaluating target and learning tracking branches properly
        self.q_eval = DeepQNetwork(input_dims, n_actions)
        self.q_eval.compile(optimizer=Adam(learning_rate=lr))
        
        self.q_next = DeepQNetwork(input_dims, n_actions)
        self.q_next.compile(optimizer=Adam(learning_rate=lr))

    def save_models(self):
        """Serializes TensorFlow weights caching physical properties logically."""
        self.q_eval.save(self.fname+'q_eval.keras')
        self.q_next.save(self.fname+'q_next.keras')
        print('... models saved successfully ...')

    def load_models(self):
        """Reconstructs historical models loading saved representations inherently."""
        self.q_eval = keras.models.load_model(self.fname+'q_eval.keras')
        self.q_next = keras.models.load_model(self.fname+'q_next.keras')
        print('... models loaded successfully ...')

    def store_transition(self, state, action, reward, state_, done):
        """Logically tracks environmental boundaries formatting history completely.

        Args:
            state (numpy.ndarray): Preceding iteration inputs generated properly.
            action (int): Generated logic decision safely mapping steps appropriately.
            reward (float): Mathematical result logically tracking algorithms safely.
            state_ (numpy.ndarray): Sequential sequence cleanly terminating branches properly.
            done (bool): Logical closure inherently preventing regressions safely.
        """
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        """Extracts numerical groupings parsing gradients inherently.

        Returns:
            tuple: PyTorch/TensorFlow compliant isolated arrays structuring calculations globally.
        """
        state, action, reward, new_state, done = \
                                  self.memory.sample_buffer(self.batch_size)
                                  
        # Cast isolated boundaries correctly binding variables universally
        states = tf.convert_to_tensor(state)
        rewards = tf.convert_to_tensor(reward)
        dones = tf.convert_to_tensor(done)
        actions = tf.convert_to_tensor(action, dtype=tf.int32)
        states_ = tf.convert_to_tensor(new_state)
        return states, actions, rewards, states_, dones

    @tf.function
    def _predict_action(self, state):
        """Compiled graph function drastically dropping native Python forward-pass overhead."""
        actions = self.q_eval(state)
        return tf.math.argmax(actions, axis=1)[0]

    def choose_action(self, observation):
        """Generates appropriate path execution applying algorithmic decay logically.

        Args:
            observation (numpy.ndarray): Visual numeric state parsed uniquely.

        Returns:
            int: Discrete bounded logic safely matching native environments properly.
        """
        if np.random.random() > self.epsilon:
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            action = self._predict_action(state).numpy()
        else:
            action = np.random.choice(self.action_space)
        return action

    def replace_target_network(self):
        """Evaluates counters triggering isolated parameter replacements effectively."""
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

    def decrement_epsilon(self):
        """Scales numeric evaluation limits generating constrained arrays dynamically."""
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    @tf.function
    def _train_step(self, states, actions, rewards, states_, dones, indices, action_indices):
        """Static computation graph dramatically bypassing sluggish Eager Execution scaling logs natively."""
        with tf.GradientTape() as tape:
            # Map precise Q-predictions corresponding cleanly to selected actions
            q_pred = tf.gather_nd(self.q_eval(states), indices=action_indices)
            q_next = self.q_next(states_)

            # Forecast limits logically executing optimal paths dynamically
            max_actions = tf.math.argmax(q_next, axis=1, output_type=tf.int32)
            max_action_idx = tf.stack([indices, max_actions], axis=1)

            # Force tensor typecast explicitly removing numpy() dependency allowing Graph mode compilation!
            dones_float = tf.cast(dones, tf.float32)
            
            # Limit future boundaries calculating standard discount rewards properly
            q_target = rewards + \
                self.gamma*tf.gather_nd(q_next, indices=max_action_idx) * \
                (1.0 - dones_float)

            # Applies isolated constraint regressions tracking targets implicitly
            loss = keras.losses.MSE(q_pred, q_target)

        # Computes localized gradients modifying arrays logically
        params = self.q_eval.trainable_variables
        grads = tape.gradient(loss, params)

        self.q_eval.optimizer.apply_gradients(zip(grads, params))

    def learn(self):
        """Calculates loss gradients executing regression tracking models natively.
        
        Derives target loss using native TensorFlow GradientTape variables safely predicting variables cleanly.
        """
        if self.memory.mem_cntr < self.batch_size:
            return

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()

        # Isolate targeted batches explicitly utilizing multi-indexing appropriately
        indices = tf.range(self.batch_size, dtype=tf.int32)
        action_indices = tf.stack([indices, actions], axis=1)

        # Execute ultra-fast Compiled Graph step instead of eager loops
        self._train_step(states, actions, rewards, states_, dones, indices, action_indices)

        self.learn_step_counter += 1
        self.decrement_epsilon()
