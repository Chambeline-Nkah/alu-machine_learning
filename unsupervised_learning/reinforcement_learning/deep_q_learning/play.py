import gymnasium as gym
from stable_baselines3 import DQN
import ale_py
from tensorflow import keras
import torch





class BreakoutPlayer:
    """
    A class to encapsulate the process of playing the Breakout game using a trained DQN model.
    """

    def __init__(self, model_path="models/policy.zip", render_mode="human"):
        """
        Initializes the player with the specified model path and rendering mode.

        Args:
            model_path (str): Path to the trained model file.
            render_mode (str): Render mode for the environment. Use "human" for visualization.
        """
        gym.register_envs(ale_py)  # Ensure ALE environments are registered
        self.env = gym.make("ALE/Breakout-v5", render_mode=render_mode)
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        """
        Loads the trained DQN model from the specified path.

        Returns:
            DQN: The loaded DQN model.
        """
        try:
            model = DQN.load(self.model_path)
            print(f"Model successfully loaded from '{self.model_path}'")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Ensure the model file exists and is correctly trained.")
            raise

    def play(self, n_episodes=5):
        """
        Runs the trained model for a specified number of episodes.

        Args:
            n_episodes (int): Number of episodes to play.
        """
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            total_reward = 0
            done = False
            steps = 0

            while not done:
                # Predict the best action using the trained model
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated

            print(
                f"Episode {episode + 1} - Total Reward: {total_reward} - Steps: {steps}"
            )

        self.env.close()

    def save_as_h5(self, output_path="models/policy.h5"):
        """
        Converts and saves the PyTorch model to HDF5 (.h5) format.

        Args:
            output_path (str): Path to save the HDF5 model.
        """
        print("Converting PyTorch model to HDF5 format...")

        # Extract the PyTorch model
        torch_model = self.model.policy.q_net

        # Convert PyTorch model to Keras-compatible format
        keras_model = keras.Sequential()
        for layer in torch_model.children():
            if isinstance(layer, torch.nn.Conv2d):
                keras_layer = keras.layers.Conv2D(
                    filters=layer.out_channels,
                    kernel_size=layer.kernel_size,
                    strides=layer.stride,
                    activation="relu",
                    padding="valid",
                    input_shape=(210, 160, 3),  # Assuming default Atari input size
                )
                keras_model.add(keras_layer)
            elif isinstance(layer, torch.nn.Linear):
                keras_layer = keras.layers.Dense(
                    units=layer.out_features,
                    activation="relu"
                    if layer.weight.shape[0] > layer.weight.shape[1]
                    else None,
                )
                keras_model.add(keras_layer)
            elif isinstance(layer, torch.nn.Flatten):
                keras_model.add(keras.layers.Flatten())

        # Save the Keras model as .h5
        keras_model.save(output_path)
        print(f"Model saved in HDF5 format: '{output_path}'")


def main():
    """
    Main function to initialize the Breakout player and run episodes.
    """
    model_path = "models/policy.zip"
    player = BreakoutPlayer(model_path=model_path)

    # Play the game for 5 episodes
    player.play(n_episodes=5)

    # Optionally save the model in HDF5 format
    save_h5 = input("Would you like to save the model in .h5 format? (y/n): ").lower()
    if save_h5 == "y":
        player.save_as_h5(output_path="models/policy.h5")


if __name__ == "__main__":
    main()