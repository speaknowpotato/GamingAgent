import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rl_env import Game2048Env
from dqn_model import DQN, ReplayBuffer
import time


class DQNAgent:
    def __init__(self):
        self.env = Game2048Env()
        self.model = DQN()
        self.target_model = DQN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.buffer = ReplayBuffer(100000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Training metrics
        self.total_steps = 0

        # 同步目标网络
        self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.model(state)
                return q_values.argmax().item()

    def train(self, episodes=1000):
        start_time = time.time()

        for episode in range(episodes):
            print(f"Episode {episode+1} / {episodes} started")
            state = self.env.reset()
            total_reward = 0
            done = False
            episode_steps = 0
            episode_loss = []
            max_tile = 0
            update_count = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                episode_steps += 1
                self.total_steps += 1

                # Track max tile on board
                current_max = max(max(row) for row in self.env.board)
                max_tile = max(max_tile, current_max)

                # 存储经验
                self.buffer.push(state, action, reward, next_state, done)

                # 训练步骤
                if len(self.buffer.buffer) >= self.batch_size:
                    loss = self._update_model()
                    episode_loss.append(loss)
                    update_count += 1

                    # Print step-level debugging info every 100 steps
                    if self.total_steps % 100 == 0:
                        print(
                            f"Step: {self.total_steps}, Loss: {loss:.4f}, Epsilon: {self.epsilon:.3f}"
                        )

                    if loss < 0.001:
                        print(f"Episode {episode+1} / {episodes} finished")
                        break
                state = next_state

            # 更新目标网络
            if episode % 10 == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            # 衰减探索率
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Calculate average loss for this episode
            avg_loss = sum(episode_loss) / len(episode_loss) if episode_loss else 0

            # Calculate elapsed time and ETA
            elapsed_time = time.time() - start_time
            avg_time_per_episode = elapsed_time / (episode + 1)
            eta = avg_time_per_episode * (episodes - episode - 1)

            # Print episode summary with detailed metrics
            print(
                f"Episode: {episode+1}/{episodes} | Steps: {episode_steps} | Updates: {update_count}"
            )
            print(
                f"Total Steps: {self.total_steps} | Max Tile: {max_tile} | Reward: {total_reward:.1f}"
            )
            print(f"Avg Loss: {avg_loss:.4f} | Epsilon: {self.epsilon:.3f}")
            print(f"Time: {elapsed_time:.1f}s | ETA: {eta:.1f}s")
            print("-" * 50)

        print(f"Training completed in {time.time() - start_time:.1f} seconds")

    def _update_model(self):
        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # 计算当前Q值
        current_q = self.model(states).gather(1, actions.unsqueeze(1))

        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # 计算损失
        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, filename="2048_dqn.pth"):
        """Save model weights"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "total_steps": self.total_steps,
            },
            filename,
        )
        print(f"Model saved to {filename}")

    def load(self, filename="2048_dqn.pth"):
        """Load model weights"""
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.target_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.total_steps = checkpoint["total_steps"]
        print(f"Model loaded from {filename}")


if __name__ == "__main__":
    agent = DQNAgent()
    try:
        agent.train(episodes=5)
        agent.save("2048_dqn.pth")
    except KeyboardInterrupt:
        print("Training interrupted by user")
        agent.save("2048_dqn_interrupted.pth")
