"""
Utility plotting and Gymnasium environment preprocessing wrappers.

This module incorporates data visualization utilities parsing Plotly metrics alongside
the essential Atari frame preprocessing `gym.Wrapper` components solving inherent
emulator flaws natively evaluating DeepMind DQN paper methodologies strictly.

Typical usage example:
    from utils import make_env, plot_learning_curve
    env = make_env('PongNoFrameskip-v4')
"""

import collections
import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gymnasium as gym

def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    """Generates an HTML Plotly graph comparing algorithm scores and epsilon trajectories dynamically.

    Constructs a dual-axis interactive webpage visualization injecting meta-refreshes
    for seamless tracking across extended chronological training epochs natively.

    Args:
        x (list): Dimensional array framing training loops natively.
        scores (list): The tracked episodic reward thresholds locally.
        epsilons (list): Chronological algorithmic tracking thresholds.
        filename (str): The logical output directory boundary natively.
        lines (list, optional): Horizontal tracking delineations structurally. Defaults to None.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=x, y=epsilons, name="Epsilon", line=dict(color="blue")),
        secondary_y=False,
    )

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        # Calculates sliding metric means averaging previous scoring brackets progressively
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])
        
    fig.add_trace(
        go.Scatter(x=x, y=running_avg, name="Running Avg Score", line=dict(color="orange")),
        secondary_y=True,
    )

    fig.update_layout(
        title_text="Learning Curve",
        xaxis_title="Training Steps"
    )

    fig.update_yaxes(title_text="Epsilon", secondary_y=False, color="blue")
    fig.update_yaxes(title_text="Score", secondary_y=True, color="orange")

    if lines is not None:
        for line in lines:
            fig.add_vline(x=line, line_width=1, line_dash="dash", line_color="red")
            
    if filename.endswith('.png'):
        filename = filename.replace('.png', '.html')
        
    fig.write_html(filename, auto_open=True, include_plotlyjs='cdn')
    
    # Inject an auto-refresh meta tag to automatically update the browser window dynamically
    try:
        with open(filename, 'r') as f:
            html_content = f.read()
            
        if '<head>' in html_content:
            html_content = html_content.replace('<head>', '<head>\n<meta http-equiv="refresh" content="5">')
            
        with open(filename, 'w') as f:
            f.write(html_content)
    except IOError:
        pass

# ====================================================================================================
class RepeatActionAndMaxFrame(gym.Wrapper):
    """Forces actions strictly evaluated consolidating repeating cycles eliminating clipping.

    Resolves hardware-level invisible-sprite artifacts present within classic Atari consoles
    by projecting strictly maximum brightness threshold constraints compiling frame sets.

    Attributes:
        repeat (int): Consecutive execution loop bounding identical inputs flexibly.
        shape (tuple): Physical bounds structuring returned observation distributions locally.
        frame_buffer (numpy.ndarray): Cache holding alternating sequence thresholds internally.
        clip_reward (bool): Triggers strict bound compression metrics normalizing states internally.
        no_ops (int): Chaotic sequential generation steps evaluating initialization natively.
        fire_first (bool): Trigger resolving logic loops requiring hardcoded initiation steps natively.
    """

    def __init__(self, env=None, repeat=4, clip_reward=False, no_ops=0,
                 fire_first=False):
        """Initializes structural tracking caches defining maximum brightness outputs."""
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros((2,) + self.shape, dtype=np.uint8)
        self.clip_reward = clip_reward
        self.no_ops = no_ops
        self.fire_first = fire_first

    def step(self, action):
        """Validates sequential iterations capturing localized threshold limits natively."""
        t_reward = 0.0
        done = False
        info = {}
        
        for i in range(self.repeat):
            obs, reward, terminated, truncated, info_ = self.env.step(action)
            done = terminated or truncated
            info.update(info_)
            if self.clip_reward:
                reward = np.clip(np.array([reward]), -1, 1)[0]
            t_reward += reward
            idx = i % 2
            self.frame_buffer[idx] = obs
            if done:
                break

        # Calculate chronological sprite maximums overriding overlapping black-screening gaps natively
        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, t_reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Restores cyclic boundaries establishing base environment dependencies correctly."""
        obs, info = self.env.reset(**kwargs)
        no_ops = np.random.randint(self.no_ops)+1 if self.no_ops > 0 else 0
        for _ in range(no_ops):
            _, _, terminated, truncated, _ = self.env.step(0)
            done = terminated or truncated
            if done:
                obs, info = self.env.reset(**kwargs)
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            obs, _, _, _, _ = self.env.step(1)

        self.frame_buffer = np.zeros((2,) + self.shape, dtype=np.uint8)
        self.frame_buffer[0] = obs

        return obs, info


# ====================================================================================================
class PreprocessFrame(gym.ObservationWrapper):
    """Processes dimensional geometric boundaries projecting normalized mathematical sequences natively.

    Compresses chaotic native dimension RGB properties standardizing uniform models properly.
    """

    def __init__(self, shape, env=None):
        """Validates standardized bounded spatial representations natively."""
        super(PreprocessFrame, self).__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                    shape=self.shape, dtype=np.float32)

    def observation(self, obs):
        """Interprets raw color spaces casting flattened scalar outputs standardly."""
        # Collapse multi-channel RGB matrix tensors cleanly evaluating isolated shapes
        new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        
        # Geometrically compress tensor bounds forcing square dimensions effectively
        resized_screen = cv2.resize(new_frame, self.shape[1:],
                                    interpolation=cv2.INTER_AREA)
        
        # Normalize structural bytes constraining numerical explosion bounds inherently
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_obs = new_obs / 255.0

        return new_obs

class StackFrames(gym.ObservationWrapper):
    """Combines sequential iterative dimensions tracking structural motion velocity globally.

    Retains consecutive frame caches mathematically combining motion tracking locally.
    """

    def __init__(self, env, repeat):
        """Instantiates double-ended queues handling memory shifts locally."""
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                            env.observation_space.low.repeat(repeat, axis=0),
                            env.observation_space.high.repeat(repeat, axis=0),
                            dtype=np.float32)
        self.stack = collections.deque(maxlen=repeat)

    def reset(self, **kwargs):
        """Overwrites stale matrices evaluating initial parameters accurately."""
        self.stack.clear()
        observation, info = self.env.reset(**kwargs)
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape), info

    def observation(self, observation):
        """Pushes trailing distributions evaluating physical sequences structurally."""
        self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)

def make_env(env_name, shape=(84,84,1), repeat=4, clip_rewards=False,
             no_ops=0, fire_first=False):
    """Compiles isolated wrapper structures linking final configurations cleanly.

    Args:
        env_name (str): The isolated Gym environment ID structuring operations correctly.
        shape (tuple, optional): Dimensions validating physical shapes. Defaults to (84, 84, 1).
        repeat (int, optional): Cyclic limits binding repetitions properly. Defaults to 4.
        clip_rewards (bool, optional): Evaluates constraint limits manually. Defaults to False.
        no_ops (int, optional): Initialization chaotic randomness bounds natively. Defaults to 0.
        fire_first (bool, optional): Specific mapping bypassing start sequences safely. Defaults to False.

    Returns:
        gym.Env: A thoroughly integrated encapsulated pipeline model structuring operations accurately.
    """
    import ale_py
    gym.register_envs(ale_py)
    
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, repeat)

    return env
