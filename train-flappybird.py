import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("agg")
import random
import numpy as np
from game.flappy_bird import Environment
import sys
import matplotlib.pyplot as plt
from agent import DQNAgent, DDQNAgent
import flappybirdutils as utils
import modelutils


def main():

    print("Creating model...")
    model = modelutils.create_model()
    model.summary()

    print("Creating agent...")
    agent_type = "ddqn"
    if agent_type == "dqn":
        agent = DQNAgent(
            name="flappybird-dqn",
            model=model,
            number_of_actions=2,
            gamma=0.99,
            final_epsilon=0.0001,
            initial_epsilon=0.1,
            number_of_iterations=2000000,
            replay_memory_size=10000,
            minibatch_size=32
        )
    elif agent_type == "ddqn":
        agent = DDQNAgent(
            name="flappybird-ddqn",
            model=model,
            number_of_actions=2,
            gamma=0.99,
            final_epsilon=0.0001,
            initial_epsilon=0.1,
            number_of_iterations=2000000,
            replay_memory_size=10000,
            minibatch_size=32,
            model_copy_interval=100
        )
    agent.enable_rewards_tracking(rewards_running_means_length=100)
    agent.enable_episodes_tracking(episodes_running_means_length=100)
    agent.enable_maxq_tracking(maxq_running_means_length=100)
    agent.enable_model_saving(model_save_frequency=100000)
    agent.enable_plots_saving(plots_save_frequency=100000)

    print("Creating game...")
    environment = Environment(headless=("headless" in sys.argv))

    print("Training ...")
    train(agent, environment, verbose="verbose" in sys.argv)


def train(agent, environment, verbose):

    # Initialize state.
    action = np.array([1.0, 0.0])
    image_data, reward, terminal = environment.step(action)
    image_data = utils.resize_and_bgr2gray(image_data)
    state = utils.image_data_to_state(image_data)

    # Main loop.
    iterations = agent.number_of_iterations
    for iteration in range(iterations):

        # Get an action. Either random or predicted. This is epsilon greedy exploration.
        action = agent.get_action(state)

        # Get next state and reward
        image_data_next, reward, terminal = environment.step(action)
        image_data_next = utils.resize_and_bgr2gray(image_data_next)
        state_next = utils.update_state(state, image_data_next)

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
