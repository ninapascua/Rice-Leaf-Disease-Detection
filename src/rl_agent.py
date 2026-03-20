# from pathlib import Path
# import json
# import random

# import matplotlib.pyplot as plt
# import numpy as np
# import tensorflow as tf
# from sklearn.metrics import accuracy_score, f1_score

# from src.data_pipeline import prepare_data


# MODEL_PATH = Path("experiments/results/best_efficientnetb0.keras")
# RL_RESULTS_PATH = Path("experiments/results/rl_results.json")
# REWARD_PLOT_PATH = Path("experiments/logs/rl_reward_curve.png")
# F1_PLOT_PATH = Path("experiments/logs/rl_f1_curve.png")
# THRESHOLD_PLOT_PATH = Path("experiments/logs/rl_threshold_curve.png")


# class ThresholdEnv:
#     def __init__(self, probs, y_true, threshold_values=None, fallback_class=0):
#         self.probs = probs
#         self.y_true = y_true
#         self.num_classes = probs.shape[1]
#         self.fallback_class = fallback_class

#         if threshold_values is None:
#             self.threshold_values = np.round(np.arange(0.30, 0.96, 0.05), 2)
#         else:
#             self.threshold_values = threshold_values

#         self.state_idx = None

#     def reset(self):
#         self.state_idx = len(self.threshold_values) // 2
#         return self.state_idx

#     def get_threshold(self):
#         return self.threshold_values[self.state_idx]

#     def step(self, action):
#         old_score = self.evaluate_threshold(self.get_threshold())

#         if action == 0:
#             self.state_idx = max(0, self.state_idx - 1)
#         elif action == 2:
#             self.state_idx = min(len(self.threshold_values) - 1, self.state_idx + 1)

#         new_threshold = self.get_threshold()
#         new_score = self.evaluate_threshold(new_threshold)

#         reward = new_score - old_score
#         done = False

#         return self.state_idx, reward, done, {
#             "threshold": float(new_threshold),
#             "macro_f1": float(new_score),
#         }

#     def evaluate_threshold(self, threshold):
#         preds = []

#         for p in self.probs:
#             top_idx = np.argmax(p)
#             top_conf = np.max(p)

#             if top_conf >= threshold:
#                 preds.append(top_idx)
#             else:
#                 preds.append(self.fallback_class)

#         preds = np.array(preds)
#         return f1_score(self.y_true, preds, average="macro")


# def plot_and_save(values, title, ylabel, path: Path):
#     path.parent.mkdir(parents=True, exist_ok=True)

#     plt.figure(figsize=(8, 5))
#     plt.plot(values)
#     plt.title(title)
#     plt.xlabel("Episode")
#     plt.ylabel(ylabel)
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.show()


# def main():
#     data = prepare_data()
#     train_df = data["train_df"]
#     val_df = data["val_df"]
#     y_val_true = data["y_val"]
#     label_encoder = data["label_encoder"]
#     class_names = data["class_names"]

#     model = tf.keras.models.load_model(MODEL_PATH)

#     X_val = []
#     for path in val_df["full_path"]:
#         img = tf.io.read_file(path)
#         img = tf.io.decode_image(img, channels=3, expand_animations=False)
#         img = tf.image.resize(img, (224, 224))
#         img = tf.cast(img, tf.float32)
#         X_val.append(img.numpy())

#     X_val = np.array(X_val)

#     val_probs = model.predict(X_val, verbose=1)
#     val_pred = np.argmax(val_probs, axis=1)

#     base_acc = accuracy_score(y_val_true, val_pred)
#     base_f1 = f1_score(y_val_true, val_pred, average="macro")

#     most_frequent_label = train_df["label"].mode()[0]
#     fallback_class = label_encoder.transform([most_frequent_label])[0]

#     env = ThresholdEnv(
#         probs=val_probs,
#         y_true=y_val_true,
#         fallback_class=fallback_class,
#     )

#     num_states = len(env.threshold_values)
#     num_actions = 3

#     Q = np.zeros((num_states, num_actions))

#     alpha = 0.1
#     gamma = 0.9
#     epsilon = 1.0
#     epsilon_decay = 0.98
#     epsilon_min = 0.05

#     episodes = 50
#     steps_per_episode = 10

#     episode_rewards = []
#     episode_thresholds = []
#     episode_f1s = []

#     for episode in range(episodes):
#         state = env.reset()
#         total_reward = 0

#         for _ in range(steps_per_episode):
#             if random.random() < epsilon:
#                 action = random.randint(0, num_actions - 1)
#             else:
#                 action = np.argmax(Q[state])

#             next_state, reward, done, info = env.step(action)

#             Q[state, action] = Q[state, action] + alpha * (
#                 reward + gamma * np.max(Q[next_state]) - Q[state, action]
#             )

#             state = next_state
#             total_reward += reward

#         epsilon = max(epsilon_min, epsilon * epsilon_decay)

#         episode_rewards.append(float(total_reward))
#         episode_thresholds.append(float(info["threshold"]))
#         episode_f1s.append(float(info["macro_f1"]))

#         print(
#             f"Episode {episode+1:02d} | "
#             f"Threshold: {info['threshold']:.2f} | "
#             f"Macro-F1: {info['macro_f1']:.4f} | "
#             f"Reward: {total_reward:.4f} | "
#             f"Epsilon: {epsilon:.4f}"
#         )

#     best_episode_idx = int(np.argmax(episode_f1s))
#     best_threshold = episode_thresholds[best_episode_idx]
#     best_f1 = episode_f1s[best_episode_idx]

#     final_state = env.reset()
#     for _ in range(10):
#         action = np.argmax(Q[final_state])
#         final_state, reward, done, info = env.step(action)

#     final_threshold = float(info["threshold"])
#     final_macro_f1 = float(info["macro_f1"])

#     RL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

#     with open(RL_RESULTS_PATH, "w") as f:
#         json.dump(
#             {
#                 "base_accuracy": float(base_acc),
#                 "base_macro_f1": float(base_f1),
#                 "best_threshold": float(best_threshold),
#                 "best_macro_f1": float(best_f1),
#                 "final_policy_threshold": float(final_threshold),
#                 "final_policy_macro_f1": float(final_macro_f1),
#                 "fallback_class": str(class_names[fallback_class]),
#             },
#             f,
#             indent=2,
#         )

#     plot_and_save(episode_rewards, "Q-Learning Reward per Episode", "Total Reward", REWARD_PLOT_PATH)
#     plot_and_save(episode_f1s, "Macro-F1 per Episode", "Macro-F1", F1_PLOT_PATH)
#     plot_and_save(episode_thresholds, "Selected Threshold per Episode", "Threshold", THRESHOLD_PLOT_PATH)

#     print(f"Saved RL results to {RL_RESULTS_PATH}")
#     print(f"Saved reward curve to {REWARD_PLOT_PATH}")
#     print(f"Saved F1 curve to {F1_PLOT_PATH}")
#     print(f"Saved threshold curve to {THRESHOLD_PLOT_PATH}")


# if __name__ == "__main__":
#     main()


from pathlib import Path
import json
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import accuracy_score, f1_score

from src.data_pipeline import prepare_data


MODEL_PATH = Path("experiments/results/best_efficientnetb0.keras")
RL_RESULTS_PATH = Path("experiments/results/rl_results.json")
REWARD_PLOT_PATH = Path("experiments/logs/rl_reward_curve.png")
F1_PLOT_PATH = Path("experiments/logs/rl_f1_curve.png")
THRESHOLD_PLOT_PATH = Path("experiments/logs/rl_threshold_curve.png")


class ThresholdEnv:
    def __init__(self, probs, y_true, threshold_values=None, fallback_class=0):
        self.probs = probs
        self.y_true = y_true
        self.num_classes = probs.shape[1]
        self.fallback_class = fallback_class

        # Include 0.00 so RL can match the original argmax classifier
        if threshold_values is None:
            self.threshold_values = np.round(np.arange(0.00, 0.96, 0.05), 2)
        else:
            self.threshold_values = threshold_values

        self.state_idx = None
        self.base_f1 = self.evaluate_threshold(0.00)

    def reset(self):
        self.state_idx = len(self.threshold_values) // 2
        return self.state_idx

    def get_threshold(self):
        return self.threshold_values[self.state_idx]

    def step(self, action):
        old_score = self.evaluate_threshold(self.get_threshold())

        if action == 0:
            self.state_idx = max(0, self.state_idx - 1)
        elif action == 2:
            self.state_idx = min(len(self.threshold_values) - 1, self.state_idx + 1)

        new_threshold = self.get_threshold()
        new_score = self.evaluate_threshold(new_threshold)

        reward = new_score - old_score
        done = False

        return self.state_idx, reward, done, {
            "threshold": float(new_threshold),
            "macro_f1": float(new_score),
        }

    def evaluate_threshold(self, threshold):
        preds = []

        for p in self.probs:
            top_idx = np.argmax(p)
            top_conf = np.max(p)

            if top_conf >= threshold:
                preds.append(top_idx)
            else:
                preds.append(self.fallback_class)

        preds = np.array(preds)
        return f1_score(self.y_true, preds, average="macro")


def plot_and_save(values, title, ylabel, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(values)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    data = prepare_data()
    train_df = data["train_df"]
    val_df = data["val_df"]
    y_val_true = data["y_val"]
    label_encoder = data["label_encoder"]
    class_names = data["class_names"]

    model = tf.keras.models.load_model(MODEL_PATH)

    X_val = []
    for path in val_df["full_path"]:
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (224, 224))
        img = tf.cast(img, tf.float32)
        img = preprocess_input(img)
        X_val.append(img.numpy())

    X_val = np.array(X_val)

    val_probs = model.predict(X_val, verbose=1)
    val_pred = np.argmax(val_probs, axis=1)

    base_acc = accuracy_score(y_val_true, val_pred)
    base_f1 = f1_score(y_val_true, val_pred, average="macro")

    most_frequent_label = train_df["label"].mode()[0]
    fallback_class = label_encoder.transform([most_frequent_label])[0]

    env = ThresholdEnv(
        probs=val_probs,
        y_true=y_val_true,
        fallback_class=fallback_class,
    )

    num_states = len(env.threshold_values)
    num_actions = 3

    Q = np.zeros((num_states, num_actions))

    alpha = 0.1
    gamma = 0.9
    epsilon = 1.0
    epsilon_decay = 0.98
    epsilon_min = 0.05

    episodes = 50
    steps_per_episode = 10

    episode_rewards = []
    episode_thresholds = []
    episode_f1s = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for _ in range(steps_per_episode):
            if random.random() < epsilon:
                action = random.randint(0, num_actions - 1)
            else:
                action = np.argmax(Q[state])

            next_state, reward, done, info = env.step(action)

            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

            state = next_state
            total_reward += reward

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        episode_rewards.append(float(total_reward))
        episode_thresholds.append(float(info["threshold"]))
        episode_f1s.append(float(info["macro_f1"]))

        print(
            f"Episode {episode+1:02d} | "
            f"Threshold: {info['threshold']:.2f} | "
            f"Macro-F1: {info['macro_f1']:.4f} | "
            f"Reward: {total_reward:.4f} | "
            f"Epsilon: {epsilon:.4f}"
        )

    best_episode_idx = int(np.argmax(episode_f1s))
    best_threshold = episode_thresholds[best_episode_idx]
    best_f1 = episode_f1s[best_episode_idx]

    final_state = env.reset()
    for _ in range(10):
        action = np.argmax(Q[final_state])
        final_state, reward, done, info = env.step(action)

    final_threshold = float(info["threshold"])
    final_macro_f1 = float(info["macro_f1"])

    RL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(RL_RESULTS_PATH, "w") as f:
        json.dump(
            {
                "base_accuracy": float(base_acc),
                "base_macro_f1": float(base_f1),
                "base_threshold": 0.0,
                "best_threshold": float(best_threshold),
                "best_macro_f1": float(best_f1),
                "final_policy_threshold": float(final_threshold),
                "final_policy_macro_f1": float(final_macro_f1),
                "fallback_class": str(class_names[fallback_class]),
            },
            f,
            indent=2,
        )

    plot_and_save(episode_rewards, "Q-Learning Reward per Episode", "Total Reward", REWARD_PLOT_PATH)
    plot_and_save(episode_f1s, "Macro-F1 per Episode", "Macro-F1", F1_PLOT_PATH)
    plot_and_save(episode_thresholds, "Selected Threshold per Episode", "Threshold", THRESHOLD_PLOT_PATH)

    print(f"Base Macro-F1 (threshold=0.00): {base_f1:.4f}")
    print(f"Best threshold found: {best_threshold:.2f}")
    print(f"Best Macro-F1: {best_f1:.4f}")
    print(f"Saved RL results to {RL_RESULTS_PATH}")


if __name__ == "__main__":
    main()