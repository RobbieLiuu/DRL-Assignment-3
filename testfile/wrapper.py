import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage


import GPUtil
torch.cuda.empty_cache()
def get_gpu_with_most_memory():
    gpus = GPUtil.getGPUs()
    max_free_mem = -1
    gpu_id_with_max_mem = -1
    for gpu in gpus:
        free_mem = gpu.memoryFree
        if free_mem > max_free_mem:
            max_free_mem = free_mem
            gpu_id_with_max_mem = gpu.id
    return gpu_id_with_max_mem, max_free_mem



import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")


# Limit the action-space to
#   0. walk right
#   1. jump right
env = JoypadSpace(env, [["right"], ["right", "A"]])

env.reset()
next_state, reward, done,  info = env.step(action=0)
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")



class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done,  info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation


# Apply Wrappers to environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)




import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torchrl.data import RandomSampler

from torchrl.data import TensorDictReplayBuffer
from torchrl.data import LazyMemmapStorage


class QNet(nn.Module):
    def __init__(self, input_channels, action_size):
        super(QNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class ICM(nn.Module):
    def __init__(self, input_channels, action_size, feature_dim=512):
        super(ICM, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, feature_dim),
            nn.ReLU()
        )

        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )

        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_size, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, state, next_state, action):
        """
        state: [batch, channels, H, W]
        next_state:  state
        action: [batch] 
        """

        phi_state = self.feature_extractor(state)
        phi_next_state = self.feature_extractor(next_state)

        inv_input = torch.cat([phi_state, phi_next_state], dim=1)
        pred_action_logits = self.inverse_model(inv_input)

        action_onehot = F.one_hot(action, num_classes=pred_action_logits.size(1)).float()

        fwd_input = torch.cat([phi_state, action_onehot], dim=1)
        pred_phi_next_state = self.forward_model(fwd_input)
        return pred_action_logits, pred_phi_next_state, phi_next_state


class DQNVariant:
    def __init__(self, state_shape, action_size, gamma=0.99, lr=1e-3,
                 buffer_size=200000, batch_size=8, tau=1e-2, update_every=10000,
                 icm_beta=0.1, forward_loss_weight=0.8):
        self.state_shape = state_shape  # (C, H, W)
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.tau = tau
        self.update_every = update_every
        self.icm_beta = icm_beta            
        self.forward_loss_weight = forward_loss_weight 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        c, h, w = state_shape
        self.q_net = QNet(c, action_size).to(self.device)
        self.target_net = QNet(c, action_size).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)

        self.icm = ICM(c, action_size, feature_dim=512).to(self.device)
        self.icm_optimizer = torch.optim.Adam(self.icm.parameters(), lr=self.lr)

        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(buffer_size, device="cpu"),
            sampler=RandomSampler(),
            batch_size=self.batch_size
        )

        self.train_step = 0

    def get_action(self, state, epsilon=0.1, deterministic=False):

        if isinstance(state, (tuple, list)):
            state = state[0]
        if not isinstance(state, (np.ndarray, list)):
            state = np.array(state)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic or random.random() > epsilon:
            with torch.no_grad():
                q_values = self.q_net(state)
            return q_values.argmax().item()
        else:
            return random.randrange(self.action_size)
        

    def add_transition(self, state, action, reward, next_state, done):

        if isinstance(state, (tuple, list)):
            state = state[0]
        if isinstance(next_state, (tuple, list)):
            next_state = next_state[0]
        
 
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)

        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)

        td = TensorDict({
            "obs": state,
            "action": torch.tensor(action, dtype=torch.int64, device="cpu"),
            "reward": torch.tensor(reward, dtype=torch.float32, device="cpu"),
            "next_obs": next_state,
            "done": torch.tensor(done, dtype=torch.bool, device="cpu")
        }, batch_size=[])
        
        self.replay_buffer.add(td)

    def update(self, soft=False):
        if soft:
            for target_param, local_param in zip(self.target_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        else:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def train(self):
        if len(self.replay_buffer) < max(self.batch_size, 1000):
            return

        batch = self.replay_buffer.sample().to(self.device) 
        states  = batch["obs"].to(self.device)  
        actions = batch["action"].long().to(self.device)  # shape: [B, 1]
        rewards = batch["reward"].to(self.device)
        next_states = batch["next_obs"].to(self.device)
        dones   = batch["done"].to(self.device)


        q_values = self.q_net(states)[range(len(actions)), actions]

        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards + torch.logical_not(dones) * self.gamma * next_q_values

   
        dqn_loss = F.smooth_l1_loss(q_values, target_q_values)


        pred_action_logits, pred_phi_next, phi_next = self.icm(states, next_states, actions)

        inverse_loss = F.cross_entropy(pred_action_logits, actions.view(-1))
        forward_loss = F.smooth_l1_loss(pred_phi_next, phi_next)

        icm_loss = self.icm_beta * inverse_loss + self.forward_loss_weight * forward_loss

        total_loss = dqn_loss + icm_loss
        total_loss = total_loss.to(self.device)

        self.optimizer.zero_grad()
        self.icm_optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.icm_optimizer.step()

        self.train_step += 1
        if self.train_step % self.update_every == 0:
            self.update(soft=False)
        torch.cuda.empty_cache()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_step': self.train_step,
            'replay_buffer': self.replay_buffer._storage.state_dict(),

            'icm': self.icm.state_dict(),
            'icm_optimizer': self.icm_optimizer.state_dict(),
        }, path)
        print(f"Model and replay buffer saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_step = checkpoint.get('train_step', 0)
        if 'replay_buffer' in checkpoint:
            self.replay_buffer._storage.load_state_dict(checkpoint['replay_buffer'])
        if 'icm' in checkpoint:
            self.icm.load_state_dict(checkpoint['icm'])
        if 'icm_optimizer' in checkpoint:
            self.icm_optimizer.load_state_dict(checkpoint['icm_optimizer'])
        print(f" Model and replay buffer loaded from {path}")

    def maybe_save_best(self, total_reward, best_reward, path="checkpoints/best_model.pt"):
        if total_reward > best_reward:
            self.save(path)
            print(f" New best model saved with reward {total_reward:.2f}")
            return total_reward
        return best_reward

#################################################################


import torch
import torch.nn as nn
import numpy as np


if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_



state_size = env.observation_space.shape
action_size = env.action_space.n
agent = DQNVariant(state_size, action_size)


num_episodes = 10000
epsilon_start = 1.0
epsilon_end = 0.0005
#epsilon_decay = 0.999
step_count = 0
epsilon = epsilon_start
reward_history = []  

checkpoint_path = "checkpoints/last_model.pt"
best_model_path = "checkpoints/best_model.pt"


best_reward = -float("inf")
if os.path.exists(checkpoint_path):
    agent.load(checkpoint_path)
    print("recover from previous checkpoint")

else:
    print("train from scratch")


for episode in range(num_episodes):
    state = env.reset() 
    done = False
    total_reward = 0

    while not done:

        action = agent.get_action(state, epsilon=epsilon)


        next_state, reward, done, _ = env.step(action)



        agent.add_transition(state, action, reward, next_state, done)


  
        agent.train()

      
        state = next_state
        total_reward += reward
        step_count += 1

    epsilon = max(epsilon_start * 10 ** (-step_count / 600000), epsilon_end)
    if (episode + 1) % 1000 == 0:
        best_reward = agent.maybe_save_best(total_reward, best_reward)
        agent.save(f"checkpoints/dqn_ep{episode+1}.pt")

    print(f"Episode {episode + 1}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")
    reward_history.append(total_reward)
