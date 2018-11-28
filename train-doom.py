import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("agg")
import random
import numpy as np
from vizdoom import *
import sys
import matplotlib.pyplot as plt
from agent import DQNAgent, DDQNAgent
import doomutils as utils
import modelutils


# These are the actions.
action_none = [0, 0, 0]
action_shoot = [0, 0, 1]
action_left = [1, 0, 0]
action_right = [0, 1, 0]
actions = [action_none, action_shoot, action_left, action_right]
action_length = len(actions)

# Parameters.
agent_type = "ddqn"


def main():

    print("Creating model...")
    model = modelutils.create_model(number_of_actions=4)
    model.summary()

    print("Creating agent...")
    if agent_type == "dqn":
        agent = DQNAgent(
            name="doom-dqn",
            model=model,
            number_of_actions=4,
            gamma=0.99,
            final_epsilon=0.0001,
            initial_epsilon=0.1,
            number_of_iterations=200000,
            replay_memory_size=10000,
            minibatch_size=32
        )
    elif agent_type == "ddqn":
        agent = DDQNAgent(
            name="doom-ddqn",
            model=model,
            number_of_actions=4,
            gamma=0.99,
            final_epsilon=0.0001,
            initial_epsilon=0.1,
            number_of_iterations=200000,
            replay_memory_size=10000,
            minibatch_size=32,
            model_copy_interval=100
        )
    agent.enable_rewards_tracking(rewards_running_means_length=1000)
    agent.enable_episodes_tracking(episodes_running_means_length=1000)
    agent.enable_maxq_tracking(maxq_running_means_length=1000)
    agent.enable_model_saving(model_save_frequency=10000)
    agent.enable_plots_saving(plots_save_frequency=10000)

    print("Creating game...")
    #environment = Environment(headless=("headless" in sys.argv))
    # Create an instance of the Doom game.
    environment = DoomGame()
    environment.load_config("scenarios/basic.cfg")
    environment.set_screen_format(ScreenFormat.GRAY8)
    environment.set_window_visible("headless" not in sys.argv)
    environment.init()

    print("Training ...")
    train(agent, environment, verbose="verbose" in sys.argv)


def train(agent, environment, verbose):

    # Main loop.
    iterations = agent.number_of_iterations
    terminal = True
    for iteration in range(iterations):

        if terminal == True:
            terminal = False
            environment.new_episode()
            image_data = environment.get_state().screen_buffer
            image_data = utils.resize_and_bgr2gray(image_data)
            state = utils.image_data_to_state(image_data)

        # Get an action. Either random or predicted. This is epsilon greedy exploration.
        action = agent.get_action(state)

        # Get next state and reward
        action_index = np.argmax(action)
        reward = environment.make_action(actions[action_index])
        terminal = environment.is_episode_finished()
        if terminal == False:
            image_data_next = environment.get_state().screen_buffer
            image_data_next = utils.resize_and_bgr2gray(image_data_next)
            state_next = utils.update_state(state, image_data_next)
        else:
            state_next = state

        # Save transition to replay memory and ensure length.
        agent.memorize_transition(state, action, reward, state_next, terminal)

        # Replay the memory.
        agent.replay_memory_via_minibatch()

        # Set state to next-state.
        state = state_next

        # Training output
        verbose = True
        if verbose:
            status_string = agent.get_status_string()
            print(status_string, end="\r")

    print("")


if __name__ == "__main__":
    main()
