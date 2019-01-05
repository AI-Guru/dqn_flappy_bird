import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("agg")
import random
import numpy as np
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_tetris
import tetrisutils as utils
import sys
import matplotlib.pyplot as plt
from agent import DQNAgent, DDQNAgent
from keras import models, layers, optimizers, initializers
import modelutils

# Parameters.
agent_type = "ddqn"
compute_custom_rewards = True
number_of_actions = 12

def main():

    print("Creating environment...")
    environment = gym_tetris.make('Tetris-v0')

    print("Creating model...")
    model = modelutils.create_model(number_of_actions)
    model.summary()

    print("Creating agent...")
    if agent_type == "dqn":
        agent = DQNAgent(
            name="tetris-dqn",
            environment=environment,
            model=model,
            observation_transformation=utils.resize_and_bgr2gray,
            observation_frames=4,
            number_of_iterations=1000000,
            gamma=0.95,
            final_epsilon=0.01,
            initial_epsilon=1.0,
            replay_memory_size=2000,
            minibatch_size=32
        )
    elif agent_type == "ddqn":
        agent = DDQNAgent(
            name="tetris-ddqn",
            environment=environment,
            model=model,
            observation_transformation=utils.resize_and_bgr2gray,
            observation_frames=4,
            number_of_iterations=1000000,
            gamma=0.95,
            final_epsilon=0.01,
            initial_epsilon=1.0,
            replay_memory_size=2000,
            minibatch_size=32,
            model_copy_interval=100
        )
    agent.enable_rewards_tracking(rewards_running_means_length=10000)
    agent.enable_episodes_tracking(episodes_running_means_length=100)
    agent.enable_maxq_tracking(maxq_running_means_length=10000)
    agent.enable_model_saving(model_save_frequency=10000)
    agent.enable_plots_saving(plots_save_frequency=10000)

    print("Training ...")
    agent.fit(verbose=True, headless="headless" in sys.argv, render_states=True)


if __name__ == "__main__":
    main()
