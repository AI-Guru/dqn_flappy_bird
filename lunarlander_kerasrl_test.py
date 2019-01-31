import warnings
warnings.filterwarnings("ignore")
import numpy as np
import gym
from keras import optimizers
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.core import Processor
import sys
from lunarlander_kerasrl_train import create_environment, build_model


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


def main():

    # Process weights filename.
    if len(sys.argv) != 2:
        print("Must provide weights file-name.")
        exit(0)
    weights_filename = sys.argv[1]

    # Create environment and model.
    environment_name = weights_filename.split("_")[2]
    environment = create_environment(environment_name)
    observation_shape = environment.observation_space.shape
    nb_actions = environment.action_space.n
    model = build_model(observation_shape, nb_actions)

    # Create memory.
    memory = SequentialMemory(limit=1000000, window_length=1) # TODO Why is this necessary?

    # Create the processor.
    #processor = CarRacingProcessor()

    # Create the DQN-Agent.
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        enable_dueling_network=True, dueling_type='avg'
        #processor=processor,
        )
    dqn.target_model = dqn.model # TODO Why is this necessary?
    dqn.compile(optimizers.Adam(lr=.00025), metrics=['mae']) # TODO Why is this necessary?

    # Load the weights.
    dqn.load_weights(weights_filename)

    # Test the agent.
    dqn.test(environment, nb_episodes=10, visualize=True)


if __name__ == "__main__":
    main()
