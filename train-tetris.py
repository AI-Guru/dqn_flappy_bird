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

    print("Creating model...")
    model = modelutils.create_model(number_of_actions)
    model.summary()

    print("Creating agent...")
    if agent_type == "dqn":
        agent = DQNAgent(
            name="tetris-dqn",
            model=model,
            number_of_actions=number_of_actions,
            gamma=0.95,
            final_epsilon=0.01,
            initial_epsilon=1.0,
            number_of_iterations=1000000,
            replay_memory_size=2000,
            minibatch_size=32
        )
    elif agent_type == "ddqn":
        agent = DDQNAgent(
            name="tetris-ddqn",
            model=model,
            number_of_actions=number_of_actions,
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
    environment = gym_tetris.make('Tetris-v0')

    print("Training ...")
    train(agent, environment, verbose="verbose" in sys.argv, headless="headless" in sys.argv)


def train(agent, environment, verbose, headless):

    # Initialize state.
    image_data = environment.reset()
    image_data = utils.resize_and_bgr2gray(image_data)
    state = utils.image_data_to_state(image_data)

    # main infinite loop
    iterations = agent.number_of_iterations
    for iteration in range(iterations):

        if headless == False:
            environment.render()

        # Get an action. Either random or predicted. This is epsilon greedy exploration.
        action = agent.get_action(state)
        action = np.argmax(action)

        # Get next state and reward
        image_data_next, reward, terminal, _ = environment.step(action)
        assert image_data_next.shape == (430, 330, 3), str(image_data_next.shape)
        image_data_next = utils.resize_and_bgr2gray(image_data_next)
        assert image_data_next.shape == (1, 84, 84), str(image_data_next.shape)
        state_next = utils.update_state(state, image_data_next)
        assert state_next.shape == (84, 84, 4), str(state_next.shape)

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
        assert state.shape == (84, 84, 4), str(state.shape)

        # Restart environment if episode is over.
        if terminal == True:
            image_data = environment.reset()
            image_data = utils.resize_and_bgr2gray(image_data)
            state = utils.image_data_to_state(image_data)

        # Training output
        verbose = True
        if verbose:
            status_string = ""
            status_string += agent.get_status_string()
            print(status_string, end="\r")

    print("")


if __name__ == "__main__":
    main()
