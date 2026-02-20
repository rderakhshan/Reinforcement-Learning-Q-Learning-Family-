import collections
import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gymnasium as gym

def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=x, y=epsilons, name="Epsilon", line=dict(color="blue")),
        secondary_y=False,
    )

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
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
        
    fig.write_html(filename, auto_open=False, include_plotlyjs='cdn')
    
    # Inject an auto-refresh meta tag to automatically update the browser window
    try:
        with open(filename, 'r') as f:
            html_content = f.read()
            
        if '<head>' in html_content:
            html_content = html_content.replace('<head>', '<head>\n<meta http-equiv="refresh" content="5">')
            
        with open(filename, 'w') as f:
            f.write(html_content)
    except IOError:
        pass

class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env=None, repeat=4, clip_reward=False, no_ops=0,
                 fire_first=False):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros((2,) + self.shape, dtype=np.uint8)
        self.clip_reward = clip_reward
        self.no_ops = no_ops
        self.fire_first = fire_first

    def step(self, action):
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

        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, t_reward, terminated, truncated, info

    def reset(self, **kwargs):
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

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env=None):
        super(PreprocessFrame, self).__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                    shape=self.shape, dtype=np.float32)

    def observation(self, obs):
        new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(new_frame, self.shape[1:],
                                    interpolation=cv2.INTER_AREA)
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_obs = new_obs / 255.0

        return new_obs

class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                            env.observation_space.low.repeat(repeat, axis=0),
                            env.observation_space.high.repeat(repeat, axis=0),
                            dtype=np.float32)
        self.stack = collections.deque(maxlen=repeat)

    def reset(self, **kwargs):
        self.stack.clear()
        observation, info = self.env.reset(**kwargs)
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape), info

    def observation(self, observation):
        self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)

def make_env(env_name, shape=(84,84,1), repeat=4, clip_rewards=False,
             no_ops=0, fire_first=False):
    import ale_py
    gym.register_envs(ale_py)
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, repeat)

    return env
