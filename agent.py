import random
import numpy as np
import time
import datetime
from keras import models
import matplotlib.pyplot as plt


class DQNAgent:

    def __init__(self, name, model, number_of_actions, gamma, final_epsilon, initial_epsilon, number_of_iterations, replay_memory_size, minibatch_size):
        self.name = name
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
        self.current_iteration = 0
        self.epsilon_decrements = np.linspace(self.initial_epsilon, self.final_epsilon, self.number_of_iterations)
        self.model_save_frequency = None

        self.track_rewards = False
        self.track_episodes = False
        self.track_maxq = False
        self.model_saving = False


    def enable_rewards_tracking(self, rewards_running_means_length):
        self.track_rewards = True
        self.rewards_running_means_length = rewards_running_means_length
        self.rewards_array = [0.0] * rewards_running_means_length
        self.current_rewards_running_means = 0
        self.rewards_running_means_array_for_plot = []


    def enable_episodes_tracking(self, episodes_running_means_length):
        self.track_episodes = True
        self.current_episode_length = 0
        self.episodes_running_means_length = episodes_running_means_length
        self.episodes_lengths_array = [0.0] * episodes_running_means_length
        self.current_episodes_running_means = 0
        self.episodes_running_means_array_for_plot = []


    def enable_maxq_tracking(self, maxq_running_means_length):
        self.track_maxq = True
        self.maxq_running_means_length = maxq_running_means_length
        self.maxq_array = [0.0] * maxq_running_means_length
        self.current_maxq_running_means = 0
        self.maxq_running_means_array_for_plot = []


    def enable_model_saving(self, model_save_frequency):
        self.model_saving = True
        self.model_save_frequency = model_save_frequency


    def enable_plots_saving(self, plots_save_frequency):
        self.plots_saving = True
        self.plots_save_frequency = plots_save_frequency


    def get_action(self, state):
        self.current_epsilon = self.epsilon_decrements[self.current_iteration]

        prediction = self.model.predict(np.expand_dims(state, axis=0))[0]
        self.current_maxq_value = np.max(prediction)

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

        # Tracking rewards.
        if self.track_rewards == True:
            self.rewards_array.append(reward)
            if len(self.rewards_array) > self.rewards_running_means_length:
                self.rewards_array.pop(0)
            self.current_rewards_running_means = np.mean(self.rewards_array)
            if self.current_iteration % self.rewards_running_means_length == 0:
                self.rewards_running_means_array_for_plot.append(self.current_rewards_running_means)

        # Tracking episode lengths.
        if self.track_episodes == True:
            self.current_episode_length += 1
            if terminal == True:
                self.episodes_lengths_array.append(self.current_episode_length)
                if len(self.episodes_lengths_array) > self.episodes_running_means_length:
                    self.episodes_lengths_array.pop(0)
                self.current_episode_length = 0
                self.current_episodes_running_means = np.mean(self.episodes_lengths_array)
                if self.current_iteration % self.episodes_running_means_length == 0:
                    self.episodes_running_means_array_for_plot.append(self.current_episodes_running_means)

        if self.track_maxq == True:
            self.maxq_array.append(self.current_maxq_value)
            if len(self.maxq_array) > self.maxq_running_means_length:
                self.maxq_array.pop(0)
            self.current_maxq_running_means = np.mean(self.maxq_array)
            if self.current_iteration % self.maxq_running_means_length == 0:
                self.maxq_running_means_array_for_plot.append(self.current_maxq_running_means)


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
        q = self.model.predict(state_batch)
        q_next = self.model.predict(state_next_batch)
        for i in range(len(minibatch)):
            if terminal_batch[i] == False:
                q[i, np.argmax(action_batch[i])] = reward_batch[i] + self.gamma * np.max(q_next[i])
            else:
                q[i, np.argmax(action_batch[i])] = reward_batch[i]

        #targets[:, np.argmax(action_batch, axis=1)] = reward_batch + agent.gamma * np.max(Q_sa, axis=1) * np.invert(terminal_batch)
        self.model.train_on_batch(state_batch, q)

        # Save the model if configured.
        if self.model_saving == True:
            self.save_model_if_enabled()

        # Save the model if configured.
        if self.plots_saving == True:
            self.save_plots_if_enabled()


    def save_model_if_enabled(self):
        # Saving the model wrt. frequency or on last iteration.
        if self.current_iteration % self.model_save_frequency == 0 or self.current_iteration == self.number_of_iterations - 1:
            self.save_model("{}-model-{:08d}.h5".format(self.name, self.current_iteration + 1))


    def save_plots_if_enabled(self):
        # Saving the model wrt. frequency or on last iteration.
        if self.current_iteration % self.plots_save_frequency == 0 or self.current_iteration == self.number_of_iterations - 1:

            if self.track_rewards:
                plt.plot(self.rewards_running_means_array_for_plot)
                plt.savefig("{}-rewards_running_means-{}.png".format(self.name, self.current_iteration + 1))
                plt.close()

            if self.track_episodes:
                plt.plot(self.episodes_running_means_array_for_plot)
                plt.savefig("{}-episodes_running_means-{}.png".format(self.name, self.current_iteration + 1))
                plt.close()

            if self.track_maxq:
                plt.plot(self.maxq_running_means_array_for_plot)
                plt.savefig("{}-maxq_running_means-{}.png".format(self.name, self.current_iteration + 1))
                plt.close()


    def save_model(self, path):
        self.model.save(path)


    def predict_on_state(self, state):
        return self.model.predict(np.expand_dims(state, axis=0))


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
        status_string += "maxq: {:.04f} ".format(self.current_maxq_value)
        if self.track_rewards:
            status_string += "rewards mean: {:.04f} ".format(self.current_rewards_running_means)
        if self.track_episodes:
            status_string += "episodes mean: {:.04f} ".format(self.current_episodes_running_means)
        if self.track_maxq:
            status_string += "maxq mean: {:.04f} ".format(self.current_maxq_running_means)
        status_string += " " * 10
        return status_string


class DDQNAgent(DQNAgent):

    def __init__(self, name, model, number_of_actions, gamma, final_epsilon, initial_epsilon, number_of_iterations, replay_memory_size, minibatch_size, model_copy_interval):
        super().__init__(name, model, number_of_actions, gamma, final_epsilon, initial_epsilon, number_of_iterations, replay_memory_size, minibatch_size)

        self.model_copy_interval = model_copy_interval
        self.target_model = models.clone_model(self.model)


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
        targets = self.target_model.predict(state_batch)
        Q_sa = self.target_model.predict(state_next_batch)
        for i in range(len(minibatch)):
            if terminal_batch[i] == False:
                targets[i, np.argmax(action_batch[i])] = reward_batch[i] + self.gamma * np.max(Q_sa[i])
            else:
                targets[i, np.argmax(action_batch[i])] = reward_batch[i]

        #targets[:, np.argmax(action_batch, axis=1)] = reward_batch + agent.gamma * np.max(Q_sa, axis=1) * np.invert(terminal_batch)
        self.model.train_on_batch(state_batch, targets)

        # Copy weights to target net.
        if self.number_of_iterations % self.model_copy_interval == 0:
            self.target_model.set_weights(self.model.get_weights())


    def save_model(self, path):
        self.target_model.save(path)


    def predict_on_state(self, state):
        return self.target_model.predict(np.expand_dims(state, axis=0))
