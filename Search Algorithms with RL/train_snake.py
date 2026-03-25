import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from snake_env import SnakeEnv
import numpy as np
import matplotlib.pyplot as plt


class StopTrainingOnEpisodes(BaseCallback):
    """
    Custom Callback:
    - Stops training after N episodes
    - Logs every episode to training_log.txt
    - Saves checkpoints at 10k, 20k, 30k, 40k, 50k, ... episodes
    """

    def __init__(self, max_episodes: int, verbose: int = 1):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.episode_count = 0

        # Prepare output log
        self.log_file = open("training_log.txt", "w")
        self.checkpoints = [10000, 20000, 30000, 40000,
                            50000, 60000, 70000, 80000, 90000, 100000]

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")

        if dones is not None:
            done_flag = bool(dones[0])

            if done_flag:
                self.episode_count += 1

                # Write to log
                log_text = f"Episode {self.episode_count} completed\n"
                self.log_file.write(log_text)
                if self.verbose:
                    print(log_text.strip())

                # Save checkpoint
                if self.episode_count in self.checkpoints:
                    ckpt_name = f"snake_model_{self.episode_count}"
                    self.model.save(ckpt_name)
                    ckpt_msg = f"[Checkpoint Saved] {ckpt_name}.zip\n"
                    self.log_file.write(ckpt_msg)
                    print(ckpt_msg)

                # Stop training if max reached
                if self.episode_count >= self.max_episodes:
                    stop_msg = f"Reached {self.episode_count} episodes. Training Complete.\n"
                    self.log_file.write(stop_msg)
                    self.log_file.close()
                    print(stop_msg)
                    return False

        return True


def save_reward_plot(model):
    """
    Creates reward curve from SB3 logs (if available)
    and saves as training_rewards.png.
    """
    try:
        rewards = model.logger.name_to_value["rollout/ep_rew_mean"]
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Mean Reward")
        plt.title("Training Reward Curve")
        plt.grid(True)
        plt.savefig("training_rewards.png")
        print("[Plot Saved] training_rewards.png")
    except:
        print("[Warning] SB3 reward logs not available for plotting.")


def train_snake(timesteps=500000, render=False):
    """Train Snake AI with PPO using a fixed number of timesteps"""

    print("Training Snake with PPO (by timesteps)")
    print(f"Total timesteps: {timesteps}")
    print(f"Render during training: {render}")
    print("-" * 40)

    # Create environment
    render_mode = "human" if render else None
    env = SnakeEnv(render_mode=render_mode)

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
    )

    # Train the model
    print("Starting training...")
    model.learn(total_timesteps=timesteps)

    # Save the model
    model.save("snake_model")
    print("Model saved as 'snake_model'")

    save_reward_plot(model)
    env.close()
    return model


def train_snake_episodes(episodes=100000, render=False):
    """Train Snake AI with PPO for a fixed number of episodes (games)."""

    print("\n============================================")
    print(f"Training Snake PPO for {episodes:,} EPISODES")
    print("============================================\n")

    render_mode = "human" if render else None
    env = SnakeEnv(render_mode=render_mode)

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
    )

    # Episode-based callback
    callback = StopTrainingOnEpisodes(max_episodes=episodes, verbose=1)

    print("Starting episode-based training...")
    # train with large timestep budget; callback will stop exactly at N episodes
    model.learn(total_timesteps=int(1e12), callback=callback)

    # Save final model
    model.save("snake_model")
    print("\n============================")
    print(" Final Model Saved: snake_model.zip")
    print("============================\n")

    save_reward_plot(model)
    env.close()
    return model


def play_trained_model(model_path="snake_model", episodes=5):
    """Watch the trained model play"""

    print(f"Loading model: {model_path}")

    env = SnakeEnv(render_mode="human")
    model = PPO.load(model_path, env=env)

    print(f"Watching trained agent play {episodes} episodes...")
    print("Close the window to stop early")

    scores = []
    for episode in range(episodes):
        obs, info = env.reset()
        done = False

        print(f"Episode {episode + 1}:")
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        score = info.get("score", 0)
        scores.append(score)
        print(f"Score: {score}")

    env.close()

    print("\nResults:")
    print(f"Average Score: {sum(scores) / len(scores):.2f}")
    print(f"Best Score: {max(scores)}")

    return scores


def main():
    import sys

    if len(sys.argv) == 1:
        train_snake()
    elif sys.argv[1] == "train":
        timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else 500000
        render = "--render" in sys.argv
        train_snake(timesteps, render)
    elif sys.argv[1] == "train_eps":
        episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
        render = "--render" in sys.argv
        train_snake_episodes(episodes, render)
    elif sys.argv[1] == "play":
        model_path = sys.argv[2] if len(sys.argv) > 2 else "snake_model"
        episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        play_trained_model(model_path, episodes)
    else:
        print("Invalid command")


if __name__ == "__main__":
    main()
