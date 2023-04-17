import torch
import torch.nn as nn
import torch.optim as optim
import random

class Env:
    def __init__(self, state_space_size):
        self.state_space_size = state_space_size
        self.reset()

    def reset(self):
        self.state = random.uniform(-1, 1)
        return self.state

    def step(self, action):
        if action == 0:
            self.state -= 0.1
        elif action == 1:
            self.state += 0.1
        else:
            raise ValueError("Invalid action")

        reward = self.compute_reward()
        done = self.is_done()
        return self.state, reward, done

    def compute_reward(self):
        if self.state >= 0.5:
            return 1
        elif self.state <= -0.5:
            return -1
        else:
            return 0

    def is_done(self):
        return abs(self.state) >= 1

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Agent:
    def __init__(self, state_space_size, action_space_size, discount_factor, learning_rate, batch_size):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.net = Net(state_space_size, action_space_size)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def get_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_space_size-1)
        else:
            with torch.no_grad():
                q_values = self.net(torch.tensor([state], dtype=torch.float32))
                return torch.argmax(q_values).item()

    def update_model(self, replay_buffer):
        batch = random.sample(replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        next_q_values = self.net(next_states).max(1)[0]
        target_q_values = rewards + self.discount_factor * next_q_values * (1 - dones)

        loss = self.criterion(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

def main(num_episodes, batch_size, learning_rate, discount_factor, epsilon_decay, min_epsilon):
    env = Env(state_space_size=1)
    agent = Agent(state_space_size=1, action_space_size=2, discount_factor=discount_factor, learning_rate=learning_rate, batch_size=batch_size)
    replay_buffer = []
    epsilon = 1.0
    total_reward = 0

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.get_action(state, epsilon)
            next_state, reward, done = env.step(action)
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward

        total_reward += episode_reward

        loss = agent.update_model(replay_buffer)

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        

    print("Average reward: {:.2f}".format(total_reward / num_episodes))


if __name__ == "__main__":
    main(num_episodes=500, batch_size=64, learning_rate=0.0005, discount_factor=0.99, epsilon_decay=0.99, min_epsilon=0.01)
