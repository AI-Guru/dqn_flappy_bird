import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import time
from game.flappy_bird import Environment
import sys
import matplotlib.pyplot as plt
from agent import Agent
import datetime
import utils

def main():
    print("Creating agent...")
    agent = Agent()
    agent.model.summary()

    print("Creating game...")
    environment = Environment(headless=("headless" in sys.argv))

    print("Training ...")
    start_time = time.time()
    train(agent, environment, start_time, verbose="verbose" in sys.argv)


def train(agent, environment, start_time, verbose):

    # initialize replay memory
    replay_memory = []

    # Initialize state.
    action = np.array([1.0, 0.0])
    image_data, reward, terminal = environment.frame_step(action)
    image_data = utils.resize_and_bgr2gray(image_data)
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

    # Saving the model.
    model_save_frequency = 100000

    # initialize epsilon value
    epsilon = agent.initial_epsilon
    iteration = 0
    epsilon_decrements = np.linspace(agent.initial_epsilon, agent.final_epsilon, agent.number_of_iterations)

    # Maximum q-values.
    max_q_values = []

    # main infinite loop
    iterations = agent.number_of_iterations
    for iteration in range(iterations):

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
        image_data_next, reward, terminal = environment.frame_step(action)
        #import scipy.misc
        #scipy.misc.imsave('outfile{}.jpg'.format(iteration), image_data_next)
        image_data_next = utils.resize_and_bgr2gray(image_data_next)

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
        Q_sa = agent.model.predict(state_next_batch)
        targets[:, np.argmax(action_batch, axis=1)] = reward_batch + agent.gamma * np.max(Q_sa, axis=1) * np.invert(terminal_batch)
        agent.model.train_on_batch(state_batch, targets)

        # Processing running means.
        rewards_array[iteration % running_mean_length] = reward
        if iteration % running_mean_frequency == 0:
            running_means.append(np.mean(rewards_array))

        # Processing q-values.
        max_q_values.append(np.max(output))

        # Set state to next-state.
        state = state_next

        # Saving the model wrt. frequency or on last iteration.
        if iteration % model_save_frequency == 0 or iteration = iterations -1:
            agent.model.save("model-{:08d}.h5".format(iteration))

            plt.plot(running_means)
            plt.savefig("running_means-{}.png".format(iteration))
            plt.close()

            plt.plot(max_q_values)
            plt.savefig("max_q_values-{}.png".format(iteration))
            plt.close()

        # Training output
        verbose = True
        if verbose:
            elapsed_time = time.time() - start_time

            estimated_time = iterations * elapsed_time / iteration - elapsed_time if iteration != 0 else 0.0

            status_string = ""
            status_string += "Progress {:.02f}% ".format(100.0 * iteration / iterations)
            status_string += "elapsed: {} ".format(str(datetime.timedelta(seconds=int(elapsed_time))))
            status_string += "estimated: {} ".format(str(datetime.timedelta(seconds=int(estimated_time))))
            status_string += "epsilon: {:.04f} ".format(epsilon)
            status_string += "action: {} ".format(action)
            status_string += "random: {} ".format(do_random_action)
            status_string += "reward: {} ".format(reward)
            status_string += "q-max: {} ".format(np.max(output))
            status_string += " " * 10
            print(status_string, end="\r")
            #print(status_string)

    print("")

    game_state.close()
    print("Done!")

    plt.plot(running_means)
    plt.savefig("running_means.png")
    plt.close()

    plt.plot(max_q_values)
    plt.savefig("max_q_values.png")
    plt.close()

    print("Really done!")





if __name__ == "__main__":
    main()
