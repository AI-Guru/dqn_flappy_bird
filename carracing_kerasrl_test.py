import warnings
warnings.filterwarnings("ignore")
import numpy as np
import gym
from keras import optimizers
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.core import Processor
import sys
from carracing_kerasrl_train import CarRacingProcessor, CarRacingDiscreteWrapper, build_model


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


def main():

    # Process weights filename.
    if len(sys.argv) != 2:
        print("Must provide weights file-name.")
        exit(0)
    weights_filename = sys.argv[1]

    # Get the environment and extract the number of actions.
    environment_name = weights_filename.split("_")[1] #"Duckietown-straight_road-v0"
    environment = gym.make(environment_name)
    environment = CarRacingDiscreteWrapper(environment)
    np.random.seed(666)
    nb_actions = environment.action_space.n

    # Build the model.
    model = build_model((WINDOW_LENGTH,) + INPUT_SHAPE, nb_actions)
    print(model.summary())

    # Create memory.
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH) # TODO Why is this necessary?

    # Create the processor.
    processor = CarRacingProcessor()

    # Create the DQN-Agent.
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        processor=processor,
        )
    dqn.target_model = dqn.model # TODO Why is this necessary?
    dqn.compile(optimizers.Adam(lr=.00025), metrics=['mae']) # TODO Why is this necessary?

    # Load the weights.
    dqn.load_weights(weights_filename)

    # Test the agent.
    dqn.test(environment, nb_episodes=10, visualize=True)


if __name__ == "__main__":
    main()
