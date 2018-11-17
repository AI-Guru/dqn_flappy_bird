import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("agg")
import random
import numpy as np
import gym
import sys
import matplotlib.pyplot as plt
from agent import Agent
from keras import models, layers, optimizers, initializers


def main():

    print("Creating model...")
    model = create_model()
    model.summary()

    print("Creating agent...")
    agent = Agent(
        model=model,
        number_of_actions=2,
        gamma=0.95,
        final_epsilon=0.01,
        initial_epsilon=1.0,
        number_of_iterations=1000000,
        replay_memory_size=2000,
        minibatch_size=32
    )
    agent.enable_running_means_tracking(100)

    print("Creating game...")
    environment = gym.make('CartPole-v0')

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

    # Initialize state.
    state = environment.reset()

    # Saving the model.
    model_save_frequency = 100000

    # Statistics.
    running_means = []
    max_q_values = []
    episode_length_means = []
    episode_length_maximums = []

    # main infinite loop
    iterations = agent.number_of_iterations

    episode_length = 0
    episode_lengths = [0]
    episode_length_maximum = 0
    for iteration in range(iterations):

        if headless == False:
            environment.render()

        # Get an action. Either random or predicted. This is epsilon greedy exploration.
        action = agent.get_action(state)

        # Get next state and reward
        state_next, reward, terminal, _ = environment.step(np.argmax(action))

        # Save transition to replay memory and ensure length.
        agent.memorize_transition(state, action, reward, state_next, terminal)

        # Update statistics.
        running_means.append(agent.current_running_means)
        max_q_values.append(agent.current_max_q_value)

        # Replay the memory.
        agent.replay_memory_via_minibatch()

        # Set state to next-state.
        state = state_next
        episode_length += 1

        if terminal == True:
            episode_lengths.append(episode_length)
            if len(episode_lengths) > 100:
                episode_lengths.pop(0)
            episode_length_maximum = max(episode_length_maximum, episode_length)
            episode_length = 0
            state = environment.reset()

        # Update statistics.
        episode_length_mean = np.mean(episode_lengths)
        episode_length_means.append(episode_length_mean)
        episode_length_maximums.append(episode_length_maximum)

        # Saving the model wrt. frequency or on last iteration.
        if iteration % model_save_frequency == 0 or iteration == iterations -1:
            agent.model.save("cartpole-model-{:08d}.h5".format(iteration + 1))

            plt.plot(running_means)
            plt.savefig("cartpole-running_means-{}.png".format(iteration + 1))
            plt.close()

            plt.plot(max_q_values)
            plt.savefig("cartpole-max_q_values-{}.png".format(iteration+ 1))
            plt.close()

            plt.plot(episode_length_means)
            plt.savefig("cartpole-episode_length_means-{}.png".format(iteration+ 1))
            plt.close()

            plt.plot(episode_length_maximums)
            plt.savefig("cartpole-episode_length_maximums-{}.png".format(iteration+ 1))
            plt.close()

        # Training output
        verbose = True
        if verbose:
            status_string = ""
            status_string += "mean: {:.01f} max: {:.01f} ".format(episode_length_mean, episode_length_maximum)

            status_string += agent.get_status_string()
            print(status_string, end="\r")

    print("")


if __name__ == "__main__":
    main()
