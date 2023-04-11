import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """
        Add an experience to the replay buffer.

        Parameters:
        - state: numpy array, state at time t
        - action: int, action taken at time t
        - reward: float, reward received at time t+1
        - next_state: numpy array, state at time t+1
        - done: boolean, indicates if episode has terminated
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the replay buffer.

        Parameters:
        - batch_size: int, size of the batch to sample

        Returns:
        - state: torch tensor, tensor of states
        - action: torch tensor, tensor of actions
        - reward: torch tensor, tensor of rewards
        - next_state: torch tensor, tensor of next states
        - done: torch tensor, tensor of done flags
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.LongTensor(action).unsqueeze(1),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent():
    def __init__(self, state_size, action_size, buffer_size, batch_size, gamma, lr):
        """
        Initialize the DQN agent.

        Parameters:
        - state_size: int, size of the state space
        - action_size: int, size of the action space
        - buffer_size: int, size of the replay buffer
        - batch_size: int, size of the training batch
        - gamma: float, discount factor for future rewards
        - lr: float, learning rate for the optimizer
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def act(self, state, epsilon):
        """
        Choose an action according to the epsilon-greedy policy.

        Parameters:
        - state
        - epsilon: float, exploration rate

        Returns:
        - action: int, chosen action
        """
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_size)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state)
            _, action = torch.max(q_values, 1)
            return int(action)

def train(self):
    """
    Train the DQN agent on a batch of experiences from the replay buffer.
    """
    if len(self.memory) < self.batch_size:
        return

    state, action, reward, next_state, done = self.memory.sample(self.batch_size)

    q_values = self.model(state).gather(1, action)
    next_q_values = self.target_model(next_state).max(1)[0].unsqueeze(1)
    expected_q_values = reward + (1 - done) * self.gamma * next_q_values

    loss = self.loss_fn(q_values, expected_q_values)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

def update_target(self):
    """
    Update the target Q-network with the current Q-network.
    """
    self.target_model.load_state_dict(self.model.state_dict())
