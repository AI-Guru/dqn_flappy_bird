import warnings
warnings.filterwarnings("ignore")
import numpy as np
import gym
import sys
import glob
from keras import models
import os
from agent import run_model

def main():
    print("Loading model...")
    if len(sys.argv) == 1:
        model_paths = glob.glob(os.path.join("pretrained_models", "*.h5"))
        model_paths = [model_path for model_path in model_paths if "cartpole" in model_path]
        model_paths = sorted(model_paths)
        model_path = model_paths[-1]
    else:
        model_path = sys.argv[1]
    model = models.load_model(model_path)
    print(model_path, "loaded.")

    print("Creating environment...")
    environment = gym.make("CartPole-v0")
    environment._max_episode_steps = 500

    print("Running ...")
    run_model(model, environment, iterations=10, observation_frames=1, observation_transformation=observation_transformation, verbose=True)


def observation_transformation(observation):
    observation = observation / np.array([2.4, 3.6, 0.27, 3.3])
    return observation


if __name__ == "__main__":
    main()
