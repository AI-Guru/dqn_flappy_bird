from PIL import Image
import numpy as np
import gym
import gym_ple
from keras import models, layers, optimizers
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.core import Processor
import sys
from flappybird_kerasrl_train import FlappyBirdProcessor

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4
env_name = "FlappyBird-v0"

def main():

    if len(sys.argv) != 2:
        print("Must provide weights file-name.")
        exit(0)

    # Get the environment and extract the number of actions.
    env = gym.make(env_name)
    np.random.seed(666)
    nb_actions = env.action_space.n

    # Next, we build our model. We use the same model that was described by Mnih et al. (2015).
    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
    model = models.Sequential()
    model.add(layers.Permute((2, 3, 1), input_shape=input_shape))
    model.add(layers.Convolution2D(32, (8, 8), strides=(4, 4), activation="relu"))
    model.add(layers.Convolution2D(64, (4, 4), strides=(2, 2), activation="relu"))
    model.add(layers.Convolution2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(nb_actions, activation="linear"))
    print(model.summary())

    memory = None
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH) # TODO Why is this necessary?
    processor = FlappyBirdProcessor()

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        processor=processor,
        )
    dqn.target_model = dqn.model # TODO Why is this necessary?
    dqn.compile(optimizers.Adam(lr=.00025), metrics=['mae']) # TODO Why is this necessary?

    weights_filename = sys.argv[1]
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True)

if __name__ == "__main__":
    main()
