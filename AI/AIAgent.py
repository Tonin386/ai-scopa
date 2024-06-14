from collections import deque
from AI.AI import AI
import torch
import random
import cupy as np
import torch.nn as nn
import torch.optim as optim

if torch.cuda.is_available():
    torch.set_default_device("cuda:0")
    torch.cuda.set_device("cuda:0")
    device = "cuda:0"

class AIAgent:
    def __init__(self, state_size, action_size, network=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        if not network:
            self.network = AI(state_size, action_size).to(device)
        else:
            self.network = network.to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, hand):
        if np.random.rand() <= self.epsilon:
            return random.choice(hand)
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        with torch.no_grad():
            act_values = self.network(state)

        act_values = np.array(act_values.cpu().numpy()[0])
        act_values = np.reshape(act_values, [10, 1])
        mask = np.zeros(self.action_size, dtype=bool)
        for i, _ in enumerate(hand):
            mask[i] = True

        act_values = act_values[mask]

        return hand[int(np.argmax(act_values))]
    
    def replay(self, game_reward=0):
        for state, action, reward, next_state, done in self.memory:
            target = reward + game_reward
            if not done:
                next_state = torch.FloatTensor(next_state).to(device).unsqueeze(0)
                with torch.no_grad():
                    max = torch.max(self.network(next_state)).item()
                    target = game_reward + reward + self.gamma * max
            
            state = torch.FloatTensor(state).to(device).unsqueeze(0)
            
            target_f = self.network(state)
            target_f = target_f.squeeze()
            target_f[action] = target
            input = self.network(state)
            input = input.squeeze()
            self.optimizer.zero_grad()
            loss = self.criterion(input.unsqueeze(0), target_f.unsqueeze(0))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_deca
        