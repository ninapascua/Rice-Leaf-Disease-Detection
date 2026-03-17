import random
import matplotlib.pyplot as plt

class RLAgent:
    def __init__(self):
        self.actions = ["increase_threshold", "decrease_threshold", "keep_threshold"]

    def select_action(self):
        return random.choice(self.actions)

    def reward(self, old_f1, new_f1):
        return new_f1 - old_f1


if __name__ == "__main__":
    agent = RLAgent()

    old_f1 = 0.25
    rewards = []

    for episode in range(10):
        action = agent.select_action()
        new_f1 = old_f1 + random.uniform(-0.03, 0.05)
        r = agent.reward(old_f1, new_f1)
        rewards.append(r)

        print(f"Episode {episode+1}")
        print("Action:", action)
        print("Old F1:", round(old_f1, 4))
        print("New F1:", round(new_f1, 4))
        print("Reward:", round(r, 4))
        print("-" * 30)

        old_f1 = new_f1

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(rewards) + 1), rewards, marker="o")
    plt.title("RL Agent Early Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.show()