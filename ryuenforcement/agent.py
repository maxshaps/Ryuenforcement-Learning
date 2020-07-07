# standard library imports
import os
import random
import math
from datetime import datetime
from collections import namedtuple
from itertools import count
from typing import Tuple, List, Union
# related third party imports
import torch
from torch import nn
import torch.nn.functional as F
from retro.retro_env import RetroEnv
import matplotlib.pyplot as plt
from IPython import display
# local application/library specific imports
from ryuenforcement.environment import get_screen
from ryuenforcement.actions import move_index_to_action_array as mi2aa


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[all_available_moves]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


def select_action(device: torch.device, state: torch.Tensor, policy_net: DQN, n_actions: int, eps_start: float,
                  eps_end: float, eps_decay: float, steps_done: int = 0) -> Tuple[torch.tensor, int]:
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * \
        math.exp(-1. * steps_done / eps_decay)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1), steps_done
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), steps_done


def optimize_model(device: torch.device, memory: ReplayMemory, policy_net: DQN, target_net: DQN, optimizer: torch.optim,
                   batch_size: int, gamma: float) -> None:
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                  dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def plot_durations(episode_durations: List[int]):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    display.clear_output(wait=True)
    display.display(plt.gcf())


def train_agent(device: torch.device, env: RetroEnv, policy_net: DQN, target_net: DQN, memory: ReplayMemory,
                optimizer: torch.optim, n_actions: int, num_episodes: int, eps_start: float, eps_end: float,
                eps_decay: float, batch_size: int, gamma: float, target_update: int, verbose: bool = True) \
        -> Tuple[List, List]:
    steps_done = 0
    episode_durations = []
    episode_rewards = []
    for i_episode in range(num_episodes):
        rewards = 0
        # Initialize the environment and state
        env.reset()
        last_screen = get_screen(device, env)
        current_screen = get_screen(device, env)
        state = current_screen - last_screen
        for t in count():
            # Select and perform an action
            # Querying policy_net for action only happens every other frame
            # No action is taken for odd numbered frames
            if t % 2 == 0:
                action, steps_done = select_action(device, state, policy_net, n_actions, eps_start, eps_end, eps_decay,
                                                   steps_done)
            else:
                action = torch.tensor([[0]], device=device, dtype=torch.long)

            _, reward, done, _ = env.step(mi2aa(action.item()))
            rewards += reward
            reward = torch.tensor([reward], device=device)
            # Observe new state
            last_screen = current_screen
            current_screen = get_screen(device, env)
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store every even numbered transition in memory
            if t % 2 == 0:
                memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            if verbose:
                if t % 100 == 0:
                    print(f'Finished {t} frames')

            # Perform one step of the optimization (on the target network)
            # Optimization only done for frames in which actions are taken
            if t % 2 == 0:
                optimize_model(device, memory, policy_net, target_net, optimizer, batch_size, gamma)
            if done:
                episode_durations.append(t + 1)
                episode_rewards.append(rewards)
                if verbose:
                    print(f'Episode {len(episode_durations)} done')
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
    return episode_durations, episode_rewards


def save_model(the_model: DQN, path: Union[None, str] = None) -> None:
    if path is None:
        current_path = os.path.dirname(os.path.abspath(__file__))
        date_str = datetime.date(datetime.now())
        path = os.path.join(current_path, 'saved_files', f'hadouken_{date_str}.pt')
        print(path)
    torch.save(the_model.state_dict(), path)
    return
