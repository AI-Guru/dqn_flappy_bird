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
agent_type = "ddqn"
compute_custom_rewards = True


def main():


    print("Creating model...")
    model = create_model()
    model.summary()

    print("Creating agent...")
    if agent_type == "dqn":
        agent = DQNAgent(
            name="cartpole-dqn",
            model=model,
            number_of_actions=2,
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
            number_of_actions=2,
            gamma=0.95,
            final_epsilon=0.01,
            initial_epsilon=1.0,
            number_of_iterations=1000000,
            replay_memory_size=2000,
            minibatch_size=32,
            model_copy_interval=100
        )
    agent.enable_rewards_tracking(rewards_running_means_length=10000)
    agent.enable_episodes_tracking(episodes_running_means_length=100)
    agent.enable_maxq_tracking(maxq_running_means_length=10000)
    agent.enable_model_saving(model_save_frequency=10000)
    agent.enable_plots_saving(plots_save_frequency=10000)

    print("Creating game...")
    environment = gym.make("CartPole-v0")
    environment._max_episode_steps = 500

    print("Training ...")
    train(agent, environment, verbose="verbose" in sys.argv, headless="headless" in sys.argv)


def create_model():
    model = models.Sequential()
    model.add(layers.Dense(24, input_dim=4, activation='relu'))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(2, activation='linear'))
    model.compile(
        loss="mse",
        optimizer=optimizers.Adam(lr=0.001))
    return model


def train(agent, environment, verbose, headless):

    # Normalization.
    observation_absolute_maximums = np.array([2.4, 3.6, 0.27, 3.3])

    # Initialize state.
    state = environment.reset()
    state = state / observation_absolute_maximums

    # main infinite loop
    iterations = agent.number_of_iterations
    for iteration in range(iterations):

        if headless == False:
            environment.render()

        # Get an action. Either random or predicted. This is epsilon greedy exploration.
        action = agent.get_action(state)

        # Get next state and reward
        state_next, reward, terminal, _ = environment.step(np.argmax(action))
        state_next = state_next / observation_absolute_maximums

        # Compute custom rewards.
        if compute_custom_rewards == True:
            if terminal == True:
                reward = -100.0
            elif np.max(np.abs(state_next)) > 0.5:
                reward = -20.0

        # Save transition to replay memory and ensure length.
        agent.memorize_transition(state, action, reward, state_next, terminal)

        # Replay the memory.
        agent.replay_memory_via_minibatch()

        # Set state to next-state.
        state = state_next

        # Restart environment if episode is over.
        if terminal == True:
            state = environment.reset()

        # Training output
        verbose = True
        if verbose:
            status_string = ""
            status_string += agent.get_status_string()
            print(status_string, end="\r")

    print("")


if __name__ == "__main__":
    main()
