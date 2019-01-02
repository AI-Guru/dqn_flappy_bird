import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("agg")
import random
import numpy as np
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym
import gym_art
import artutils as utils
import sys
import matplotlib.pyplot as plt
from agent import *
from keras import models, layers, optimizers, initializers

# Parameters.
frames = 4
compute_custom_rewards = True
number_of_actions = 12

def main():

    print("Creating environment...")
    environment = gym.make("art-mnist-v0")

    print("Creating model...")
    input_dimensions = [space.shape for space in environment.observation_space.spaces][0]
    output_dimensions = [space.n for space in environment.action_space.spaces]
    model = utils.create_model(input_frames=frames, input_dimensions=input_dimensions, output_dimensions=output_dimensions, cnn_blocks=2, dense_dimensions=[])
    model.summary()

    print("Creating agent...")
    agent = DQNAgentSpecial(
        name="art-dqn",
        model=model,
        gamma=0.95,
        final_epsilon=0.01,
        initial_epsilon=1.0,
        number_of_iterations=100000,
        replay_memory_size=2000,
        minibatch_size=32,
        model_copy_interval=256
    )

    agent.enable_rewards_tracking(rewards_running_means_length=100)
    agent.enable_episodes_tracking(episodes_running_means_length=100)
    agent.enable_maxq_tracking(maxq_running_means_length=1000)
    agent.enable_model_saving(model_save_frequency=1000)
    agent.enable_plots_saving(plots_save_frequency=1000)

    print("Training ...")
    train(agent, environment, verbose="verbose" in sys.argv, headless="headless" in sys.argv)


def train(agent, environment, verbose, headless):

    # Initialize state.
    (observation_target, observation_canvas) = environment.reset()
    state = utils.image_data_to_state(observation_target, observation_canvas, frames)

    # main infinite loop
    iterations = agent.number_of_iterations
    for iteration in range(iterations):

        if headless == False:
            environment.render()

        # Get an action. Either random or predicted. This is epsilon greedy exploration.
        action = agent.get_action(state)
        action = [np.argmax(a) for a in action]

        # Get next state and reward
        (_, observation_canvas_next), reward, terminal, _ = environment.step(action)
        state_next = utils.update_state(state, observation_canvas_next)

        # Save transition to replay memory and ensure length.
        agent.memorize_transition(state, action, reward, state_next, terminal)

        # Replay the memory.
        agent.replay_memory_via_minibatch()

        # Set state to next-state.
        state = state_next

        # Restart environment if episode is over.
        if terminal == True:
            (observation_target, observation_canvas) = environment.reset()
            state = utils.image_data_to_state(observation_target, observation_canvas, frames)

        # Training output
        verbose = True
        if verbose:
            status_string = ""
            status_string += agent.get_status_string()
            print(status_string, end="\r")

    print("")


if __name__ == "__main__":
    main()
