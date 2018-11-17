import random
import numpy as np
import time
import datetime


class Agent:

    def __init__(self, model, number_of_actions, gamma, final_epsilon, initial_epsilon, number_of_iterations, replay_memory_size, minibatch_size):
        self.model = model
        self.number_of_actions = number_of_actions
        self.gamma = gamma
        self.final_epsilon = final_epsilon
        self.initial_epsilon = initial_epsilon
        self.number_of_iterations = number_of_iterations
        self.replay_memory_size = replay_memory_size
        self.minibatch_size = minibatch_size

        # BLA
        self.start_time = time.time()
        self.replay_memory = []
        #self.epsilon = self.initial_epsilon
        self.current_iteration = 0
        self.epsilon_decrements = np.linspace(self.initial_epsilon, self.final_epsilon, self.number_of_iterations)


    def enable_running_means_tracking(self, running_mean_length):
        self.track_running_means = True
        self.running_mean_length = running_mean_length
        self.rewards_array = np.zeros((running_mean_length, ))


    def get_action(self, state):
        self.current_epsilon = self.epsilon_decrements[self.current_iteration]

        prediction = self.model.predict(np.expand_dims(state, axis=0))[0]
        self.current_max_q_value = np.max(prediction)

        do_random_action = random.random() <= self.current_epsilon
        if do_random_action:
            action_index = random.randint(0, self.number_of_actions - 1)
        else:
            action_index = np.argmax(prediction)
        action = np.zeros((self.number_of_actions,)).astype("float32")
        action[action_index] = 1.0

        self.current_action = action

        return action


    def memorize_transition(self, state, action, reward, state_next, terminal):
        self.replay_memory.append((state, action, reward, state_next, terminal))
        if len(self.replay_memory) > self.replay_memory_size:
            self.replay_memory.pop(0)

        self.current_reward = reward
        self.current_iteration += 1

        if self.track_running_means == True:
            self.rewards_array[self.current_iteration % self.running_mean_length] = reward
            self.current_running_means = np.mean(self.rewards_array)


    def replay_memory_via_minibatch(self):

        # sample random minibatch
        minibatch = random.sample(self.replay_memory, min(len(self.replay_memory), self.minibatch_size))

        # unpack minibatch
        state_batch = np.array([d[0] for d in minibatch])
        action_batch = np.array([d[1] for d in minibatch])
        reward_batch = np.array([d[2] for d in minibatch])
        state_next_batch = np.array([d[3] for d in minibatch])
        terminal_batch = np.array([d[4] for d in minibatch])

        # Do gradient descent.
        targets = self.model.predict(state_batch)
        Q_sa = self.model.predict(state_next_batch)
        for i in range(len(minibatch)):
            if terminal_batch[i] == False:
                targets[i, np.argmax(action_batch[i])] = reward_batch[i] + self.gamma * np.max(Q_sa[i])
            else:
                targets[i, np.argmax(action_batch[i])] = reward_batch[i]

        #targets[:, np.argmax(action_batch, axis=1)] = reward_batch + agent.gamma * np.max(Q_sa, axis=1) * np.invert(terminal_batch)
        self.model.train_on_batch(state_batch, targets)

    def get_status_string(self):

        elapsed_time = time.time() - self.start_time
        estimated_time = self.number_of_iterations * elapsed_time / self.current_iteration - elapsed_time if self.current_iteration != 0 else 0.0

        status_string = ""
        status_string += "{}/{} " .format(self.current_iteration, self.number_of_iterations)
        status_string += "{:.02f}% ".format(100.0 * self.current_iteration / self.number_of_iterations)
        status_string += "elapsed: {} ".format(str(datetime.timedelta(seconds=int(elapsed_time))))
        status_string += "estimated: {} ".format(str(datetime.timedelta(seconds=int(estimated_time))))
        status_string += "epsilon: {:.04f} ".format(self.current_epsilon)
        status_string += "action: {} ".format(self.current_action)
        #status_string += "random: {} ".format(do_random_action)
        status_string += "reward: {} ".format(self.current_reward)
        status_string += "q-max: {:.04f} ".format(self.current_max_q_value)
        if self.track_running_means:
            status_string += "running means: {:.04f} ".format(self.current_running_means)
        status_string += " " * 10
        return status_string
