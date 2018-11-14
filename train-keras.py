
# Make then thing headless.
# TODO put this into the game
import os
#os.putenv('SDL_VIDEODRIVER', 'dummy')

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from game.flappy_bird import GameState

import sys
import matplotlib.pyplot as plt

from keras import models, layers, optimizers, initializers

class Agent:

    def __init__(self):
        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 20000 #2000000
        self.replay_memory_size = 10000
        self.minibatch_size = 32

        self.model = self.create_model()

    def create_model(self):
        kernel_initializer = initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=None)
        bias_initializer = initializers.Constant(value=0.01)

        model = models.Sequential()
        model.add(layers.Conv2D(32, (8, 8), strides=(4, 4), activation="relu", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, input_shape=(84, 84, 4)))
        model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), activation="relu", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation="relu", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation="relu", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(layers.Dense(self.number_of_actions, activation="linear", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.compile(
            optimizer=optimizers.Adam(lr=1e-6),
            loss="mse"
        )
        return model

Agent().model.summary()

def resize_and_bgr2gray(image):
    image = image[0:288, 0:404]
    image = color.rgb2gray(image)
    image = resize(image, (84, 84), anti_aliasing=True)
    image = np.reshape(image, (84, 84, 1))
    image[image > 0] = 255
    image = image.transpose(2, 0, 1)
    image = image.astype(np.float32) / 255.0 # TODO division right?
    return image

def render_state(state):
    for i in range(state.shape[-1]):
        image_data = state[0,:,:, i]
        print(image_data.shape)
        plt.subplot(1, state.shape[-1], i + 1)
        plt.imshow(image_data, cmap="gray")

    plt.show()
    plt.close()


def train(agent, start_time):

    # instantiate game
    game_state = GameState()

    # initialize replay memory
    replay_memory = []

    # initial action is do nothing
    action = np.zeros((agent.number_of_actions,)).astype("float32")
    action[0] = 1.0

    # Initialize state.
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    state = np.zeros((84, 84, 4))
    state[:,:,0] = image_data
    state[:,:,1] = image_data
    state[:,:,2] = image_data
    state[:,:,3] = image_data

    # Initialize running means.
    running_mean_length = 1000
    running_mean_frequency = 100
    rewards_array = np.zeros((running_mean_length, ))
    running_means = []

    # initialize epsilon value
    epsilon = agent.initial_epsilon
    iteration = 0
    epsilon_decrements = np.linspace(agent.initial_epsilon, agent.final_epsilon, agent.number_of_iterations)

    # main infinite loop
    for iteration in range(agent.number_of_iterations):

        # Get an action. Either random or predicted. This is epsilon greedy exploration.
        epsilon = epsilon_decrements[iteration]
        action = np.zeros((agent.number_of_actions,)).astype("float32")
        do_random_action = random.random() <= epsilon
        output = agent.model.predict(np.expand_dims(state, axis=0))[0]
        if do_random_action:
            action_index = random.randint(0, agent.number_of_actions - 1)
        else:
            action_index = np.argmax(output)
        action[action_index] = 1.0

        # Get next state and reward
        image_data_next, reward, terminal = game_state.frame_step(action)
        #import scipy.misc
        #print(image_data_next.shape)
        #scipy.misc.imsave('outfile{}.jpg'.format(iteration), image_data_next)
        image_data_next = resize_and_bgr2gray(image_data_next)
        #scipy.misc.imsave('outfile-bw.png', image_data_next)

        #state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)
        state_next = np.zeros((84, 84, 4))
        state_next[:,:,0] = state[:,:,1]
        state_next[:,:,1] = state[:,:,2]
        state_next[:,:,2] = state[:,:,3]
        state_next[:,:,3] = image_data_next
        #render_state(state_1)

        #reward = np.array([reward], dtype=np.float32)
        #reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        # Save transition to replay memory and ensure length.
        replay_memory.append((state, action, reward, state_next, terminal))
        if len(replay_memory) > agent.replay_memory_size:
            replay_memory.pop(0)

        #print("replay_memory", len(replay_memory))

        # sample random minibatch
        minibatch = random.sample(replay_memory, min(len(replay_memory), agent.minibatch_size))

        # unpack minibatch
        state_batch = np.array([d[0] for d in minibatch])
        action_batch = np.array([d[1] for d in minibatch])
        reward_batch = np.array([d[2] for d in minibatch])
        state_next_batch = np.array([d[3] for d in minibatch])
        terminal_batch = np.array([d[4] for d in minibatch])

        # Do gradient descent.
        targets = agent.model.predict(state_batch)
        #print("targets:", targets)

        Q_sa = agent.model.predict(state_next_batch)
        #print("Q_sa:   ", Q_sa)
        #print("argmax: ", np.argmax(action_batch, axis=1))

        targets[:, np.argmax(action_batch, axis=1)] = reward_batch + agent.gamma * np.max(Q_sa, axis=1) * np.invert(terminal_batch)
        #print("reward_batch", reward_batch)
        #print("targets:", targets)

        agent.model.train_on_batch(state_batch, targets)
        #print("")

        rewards_array[iteration % running_mean_length] = reward
        if iteration % running_mean_frequency == 0:
            running_means.append(np.mean(rewards_array))

        # Set state to next-state.
        state = state_next

        # Training output
        verbose = True
        if verbose:
            status_string = ""
            status_string += "Progress {:.02f}% ".format(100.0 * iteration / agent.number_of_iterations)
            status_string += "time: {:.02f}s ".format(time.time() - start_time)
            status_string += "epsilon: {:.04f} ".format(epsilon)
            status_string += "action: {} ".format(action)
            status_string += "random: {} ".format(do_random_action)
            status_string += "reward: {} ".format(reward)
            status_string += "q-max: {} ".format(np.max(output))
            print(status_string, end="\r")

    print("")
    print(len(running_means))
    plt.plot(running_means)
    plt.show()
    plt.close()

agent = Agent()
start_time = time.time()
train(agent, start_time)

vdisplay.stop()
