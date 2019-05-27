import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import numpy as np
import gym
from keras import models, layers, optimizers
from rl.agents import NAFAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from kerasrl_extensions import *
import sys

# Turning off the logger.
import logging
logger = logging.getLogger("gym-duckietown")
logger.propagate = False


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4
#environment_name = "Duckietown-straight_road-v0"
environment_name = "CarRacing-v0"


class CarRacingProcessor(Processor):
    """
    Transforms observations, state-batches and rewards of
    Flappy-Bird.
    """

    def process_observation(self, observation):
        """
        Takes an observation, resizes it and turns it into greyscale.
        """
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')
        processed_observation = np.array(img)
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        """
        Normalizes a batch of observations.
        """
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        """
        Clips the rewards.
        """
        return reward
        #return np.clip(reward, -1., 1.)

    def process_info(self, info):
        """
        Ignores the info.
        """
        return {}


def train(index, policy_nb_steps, fit_nb_steps):

    # Get the environment and extract the number of actions.
    print("Using environment", environment_name)
    environment = gym.make(environment_name)
    np.random.seed(666)
    nb_actions =  environment.action_space.shape[0]

    # Build the model.
    v_model, mu_model, l_model = build_models((WINDOW_LENGTH,) + INPUT_SHAPE, nb_actions)
    v_model.summary()
    mu_model.summary()
    l_model.summary()

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    processor = CarRacingProcessor()
    random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, size=nb_actions)

    agent = NAFAgent(nb_actions=nb_actions, V_model=v_model, L_model=l_model, mu_model=mu_model,
                 memory=memory, nb_steps_warmup=100, random_process=random_process,
                 gamma=.99, target_model_update=1e-3, processor=processor)
    agent.compile(optimizers.Adam(lr=.001, clipnorm=1.), metrics=['mae'])

    weights_filename = 'naf_{}_{}_weights.h5f'.format(environment_name, index)

    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that now you can use the built-in Keras callbacks!
    checkpoint_weights_filename = 'naf_' + environment_name + '_weights_{step}.h5f'
    log_filename = 'naf_{}_log.json'.format(environment_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [TensorboardCallback()]
    callbacks += [FileLogger(log_filename, interval=100)]
    agent.fit(
        environment,
        callbacks=callbacks,
        #nb_steps=1750000,
        nb_steps=fit_nb_steps,
        log_interval=10000,
        visualize="visualize" in sys.argv)

    # After training is done, we save the final weights one more time.
    agent.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    #dqn.test(environment, nb_episodes=10, visualize=False)

def build_models(input_shape, actions):

    cnn_model = models.Sequential()
    cnn_model.add(layers.Permute((2, 3, 1), input_shape=input_shape))
    cnn_model.add(layers.Convolution2D(32, (8, 8), strides=(4, 4), activation="relu"))
    cnn_model.add(layers.Convolution2D(64, (4, 4), strides=(2, 2), activation="relu"))
    cnn_model.add(layers.Convolution2D(64, (3, 3), strides=(1, 1), activation="relu"))
    cnn_model.add(layers.Flatten())

    v_model_input = layers.Input(shape=input_shape)
    v_model_output = cnn_model(v_model_input)
    v_model_output = layers.Dense(512, activation="relu")(v_model_output)
    v_model_output = layers.Dense(1, activation="linear")(v_model_output)
    v_model = models.Model(v_model_input, v_model_output)

    mu_model_input = layers.Input(shape=input_shape)
    mu_model_output = cnn_model(mu_model_input)
    mu_model_output = layers.Dense(512, activation="relu")(mu_model_output)
    mu_model_output = layers.Dense(actions, activation="linear")(mu_model_output)
    mu_model = models.Model(mu_model_input, mu_model_output)

    l_model_action_input = layers.Input(shape=(actions,))
    l_model_action_output = l_model_action_input
    l_model_observation_input = layers.Input(shape=input_shape)
    l_model_observation_output = cnn_model(l_model_observation_input)
    l_model_output = layers.Concatenate()([l_model_action_output, l_model_observation_output])
    l_model_output = layers.Dense(512, activation="relu")(l_model_output)
    l_model_output = layers.Dense(((actions * actions + actions) // 2), activation="linear")(l_model_output)
    l_model = models.Model([l_model_action_input, l_model_observation_input], l_model_output)

    return v_model, mu_model, l_model


if __name__ == "__main__":
    parameters = [
        (4000, 5000),
        (40000, 50000),
        (400000, 500000),
    ]
    for index, (policy_nb_steps, fit_nb_steps) in enumerate(parameters):
        train(index, policy_nb_steps, fit_nb_steps)
