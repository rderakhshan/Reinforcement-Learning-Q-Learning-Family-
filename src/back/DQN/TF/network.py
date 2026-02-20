"""
Deep Q-Network architecture implementation using TensorFlow/Keras.

This module provides the neural network architecture used by the TensorFlow DQN agent.
It implements the precise 3-layer convolutional network proposed in the
original DeepMind paper (Mnih et al., 2015) for processing Atari game frames natively.

Typical usage example:
    from network import DeepQNetwork
    net = DeepQNetwork(input_dims=(4, 84, 84), n_actions=6)
"""

import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten


class DeepQNetwork(keras.Model):
    """Convolutional Neural Network for Deep Q-Learning (TensorFlow Keras).

    Implements the core CNN architecture that evaluates pixel data and outputs 
    approximated Q-values utilizing native TensorFlow layers natively.

    Attributes:
        conv1 (tensorflow.keras.layers.Conv2D): First scaling layer natively (32 channels).
        conv2 (tensorflow.keras.layers.Conv2D): Second scaling layer natively (64 channels).
        conv3 (tensorflow.keras.layers.Conv2D): Third evaluating layer natively (64 channels).
        flat (tensorflow.keras.layers.Flatten): Routs multi-channel frames geometrically.
        fc1 (tensorflow.keras.layers.Dense): Fully connected Dense grouping logic logically.
        fc2 (tensorflow.keras.layers.Dense): Terminal prediction matrix returning absolute variables.
    """

    def __init__(self, input_dims, n_actions):
        """Initializes the TensorFlow Keras network sequentially parameters strictly.

        Args:
            input_dims (tuple): Absolute tensor constraints bounded natively.
            n_actions (int): Available logical actions mapping numerical sequences logically.
        """
        super(DeepQNetwork, self).__init__()
        
        self.input_dims = input_dims
        self.n_actions = n_actions
        
        # Geometrically scale frames utilizing channels_last (NHWC) explicitly allowing CPU processing
        self.conv1 = Conv2D(32, 8, strides=(4, 4), activation='relu')
        self.conv2 = Conv2D(64, 4, strides=(2, 2), activation='relu')
        self.conv3 = Conv2D(64, 3, strides=(1, 1), activation='relu')
                            
        self.flat = Flatten()
        
        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(n_actions, activation=None)

    def get_config(self):
        """Allows Keras to serialize tracking parameters cleanly internally.

        Returns:
            dict: The localized dictionary bounding tracking logic dynamically.
        """
        config = super(DeepQNetwork, self).get_config()
        config.update({
            "input_dims": self.input_dims,
            "n_actions": self.n_actions,
        })
        return config

    def call(self, state):
        """Executes Keras feed-forward evaluation mathematically.

        Args:
            state (tf.Tensor): Batched inputs defining structural evaluation cleanly.

        Returns:
            tf.Tensor: Array outputs evaluating choices logically.
        """
        # Transpose batched arrays from PyTorch format (N, C, H, W) to CPU-safe TensorFlow format (N, H, W, C)
        import tensorflow as tf
        x = tf.transpose(state, perm=[0, 2, 3, 1])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x
