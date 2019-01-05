import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("agg")
import random
import numpy as np
import gym
import sys
import matplotlib.pyplot as plt
from agent import DQNAgent, DDQNAgent
from keras import models, layers, optimizers, initializers


# Parameters.
agent_type = "dqn"
compute_custom_rewards = True


def main():

    print("Creating model...")
    model = create_model()
    model.summary()

    print("Creating environment...")
    environment = gym.make("CartPole-v0")
    environment._max_episode_steps = 500

    print("Creating agent...")
    if agent_type == "dqn":
        agent = DQNAgent(
            name="cartpole-dqn",
            model=model,
            environment=environment,
            observation_frames=1,
            observation_transformation=observation_transformation,
            reward_transformation=reward_transformation,
            gamma=0.95,
            final_epsilon=0.01,
            initial_epsilon=1.0,
            number_of_iterations=1000000,
            replay_memory_size=2000,
            minibatch_size=32
        )
    elif agent_type == "ddqn":
        agent = DDQNAgent(
            name="cartpole-ddqn",
            model=model,
            environment=environment,
            observation_frames=1,
            observation_transformation=observation_transformation,
            reward_transformation=reward_transformation,
            gamma=0.95,
            final_epsilon=0.01,
            initial_epsilon=1.0,
            number_of_iterations=1000000,
            replay_memory_size=2000,
            minibatch_size=32,
            model_copy_interval=100
        )
    agent.enable_rewards_tracking(rewards_running_means_length=10000)
    agent.enable_episodes_tracking(episodes_running_means_length=10000)
    agent.enable_maxq_tracking(maxq_running_means_length=10000)
    agent.enable_model_saving(model_save_frequency=100000)
    agent.enable_tensorboard_for_tracking()

    print("Training ...")
    agent.fit(verbose=True, headless="render" not in sys.argv)


def create_model():
    model = models.Sequential()
    model.add(layers.Dense(24, input_dim=4, activation='relu'))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(2, activation='linear'))
    model.compile(
        loss="mse",
        optimizer=optimizers.Adam(lr=0.001))
    return model


def observation_transformation(observation):
    observation = observation / np.array([2.4, 3.6, 0.27, 3.3])
    return observation


def reward_transformation(state, reward, terminal):
    if terminal == True:
        reward = -100.0
    elif np.max(np.abs(state)) > 0.5:
        reward = -20.0
    return reward


if __name__ == "__main__":
    main()
