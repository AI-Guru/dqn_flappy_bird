import random
import numpy as np
import time
import datetime
from keras import models
import tensorflow as tf
import matplotlib.pyplot as plt


class DQNAgent:
    """
    This is the implementation of an DQN-Agent.

    It allows for doing deep reinforcement learning. It also does
    minibatch-replay against catastrophic forgetting.
    """

    def __init__(self, name, model, number_of_actions, gamma, final_epsilon, initial_epsilon, number_of_iterations, replay_memory_size, minibatch_size):
        """
        Creates a DQN-agent.

        Args:
            name (string): Name of the agent.
            model (keras model): A neural network.
            number_of_actions (int): Number of actions the agent can perform.
            gamma (float): Weight factor for future rewards.
            final_epsilon (float): Final value for epsilon-greedy.
            initial_epsilon (float): Initial value for epsilon-greedy.
            number_of_iterations (int): How many iterations to run.
            replay_memory_size (int): Size of the memory for replay.
            minibatch_size (int): Size of the minibatches used during replay.
        """

        # Constructor parameters.
        self.name = name
        self.model = model
        self.number_of_actions = number_of_actions
        self.gamma = gamma
        self.final_epsilon = final_epsilon
        self.initial_epsilon = initial_epsilon
        self.number_of_iterations = number_of_iterations
        self.replay_memory_size = replay_memory_size
        self.minibatch_size = minibatch_size

        # Additional variables.
        self.start_time = time.time()
        self.replay_memory = []
        self.current_iteration = 0
        self.epsilon_decrements = np.linspace(self.initial_epsilon, self.final_epsilon, self.number_of_iterations)
        self.model_save_frequency = None

        # Additional functions. See below.
        self.track_rewards = False
        self.track_episodes = False
        self.track_maxq = False
        self.save_model_automatically = False
        self.save_plots_automatically = False


    def enable_rewards_tracking(self, rewards_running_means_length):
        """
        Enables tracking the rewards during training.

        Uses a running mean.

        Args:
            rewards_running_means_length (int): Amount of values used for computing the running mean.
        """

        self.track_rewards = True
        self.rewards_running_means_length = rewards_running_means_length
        self.rewards_array = [0.0] * rewards_running_means_length
        self.current_rewards_running_means = 0
        self.rewards_running_means_array_for_plot = []


    def enable_episodes_tracking(self, episodes_running_means_length):
        """
        Enables tracking the episodes during training.

        Uses a running mean.

        Args:
            episodes_running_means_length (int): Amount of values used for computing the running mean.
        """

        self.track_episodes = True
        self.current_episode_length = 0
        self.episodes_running_means_length = episodes_running_means_length
        self.episodes_lengths_array = [0.0] * episodes_running_means_length
        self.current_episodes_running_means = 0
        self.episodes_running_means_array_for_plot = []


    def enable_maxq_tracking(self, maxq_running_means_length):
        """
        Enables tracking the max-q-values during training.

        Uses a running mean.

        Args:
            maxq_running_means_length (int): Amount of values used for computing the running mean.
        """

        self.track_maxq = True
        self.maxq_running_means_length = maxq_running_means_length
        self.maxq_array = [0.0] * maxq_running_means_length
        self.current_maxq_running_means = 0
        self.maxq_running_means_array_for_plot = []


    def enable_tensorboard_for_tracking(self):
        """
        Enables tensorboard for tracking.
        """
        self.tensorboard_writer = tf.summary.FileWriter("tensorboard")


    def _log_scalar(self, tag, value, step):
        tag += " (" + self.name + ")"
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.tensorboard_writer.add_summary(summary, step)


    def enable_model_saving(self, model_save_frequency):
        """
        Enables saving the model during training.

        Args:
            model_save_frequency (int): Frequency for saving the model automatically.
        """

        self.save_model_automatically = True
        self.model_save_frequency = model_save_frequency


    def enable_plots_saving(self, plots_save_frequency):
        """
        Enables automatic plotting of key-values.

        Args:
            plots_save_frequency (int): Frequency for plotting the values automatically.
        """

        self.save_plots_automatically = True
        self.plots_save_frequency = plots_save_frequency


    def get_action(self, state):
        """
        Returns an action.

        This implements epsilon greedy. That is, it returns with a certain probability
        either a random action or a predicted action.

        Args:
            state (ndarray): A state of the environment.

        Returns:
            ndarray: A one-hot-encoded action.
        """

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
        """
        Stores a transition in memory.

        This is necessary for minibatch-replay. Also keeps track of statistics
        if enabled.

        A bit longer description.

        Args:
            state (ndarray): The current state of the environment.
            action (nd.array): One-hot encoded action.
            reward (float): Reward from the environment base on the action.
            state_next (ndarray): The next state of the environment.
            terminal (boolean): Denotes if the state is terminal or not.
        """

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
                if self.tensorboard_writer:
                    self._log_scalar("Rewards running means", self.current_rewards_running_means, self.current_iteration)

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
                if self.tensorboard_writer:
                    self._log_scalar("Episodes running means", self.current_episodes_running_means, self.current_iteration)

        # Tracking max-q values.
        if self.track_maxq == True:
            self.maxq_array.append(self.current_maxq_value)
            if len(self.maxq_array) > self.maxq_running_means_length:
                self.maxq_array.pop(0)
            self.current_maxq_running_means = np.mean(self.maxq_array)
            if self.current_iteration % self.maxq_running_means_length == 0:
                self.maxq_running_means_array_for_plot.append(self.current_maxq_running_means)
                if self.tensorboard_writer:
                    self._log_scalar("Max-Q running means", self.current_maxq_running_means, self.current_iteration)


    def replay_memory_via_minibatch(self):
        """
        Replays memory in one minibatch.

        Used to prevent catastrophic forgetting. Also saves model and plots,
        if previously activated.
        """

        # Sample random minibatch and unpack it.
        minibatch = random.sample(self.replay_memory, min(len(self.replay_memory), self.minibatch_size))
        state_batch = np.array([d[0] for d in minibatch])
        action_batch = np.array([d[1] for d in minibatch])
        reward_batch = np.array([d[2] for d in minibatch])
        state_next_batch = np.array([d[3] for d in minibatch])
        terminal_batch = np.array([d[4] for d in minibatch])

        # Prepare q-values.
        q_values = self.model.predict(state_batch)
        q_values_next = self.model.predict(state_next_batch)
        for i in range(len(minibatch)):
            if terminal_batch[i] == False:
                q_values[i, np.argmax(action_batch[i])] = reward_batch[i] + self.gamma * np.max(q_values_next[i])
            else:
                q_values[i, np.argmax(action_batch[i])] = reward_batch[i]

        # Do gradient descent.
        self.model.train_on_batch(state_batch, q_values)

        # Save the model if configured.
        if self.save_model_automatically == True:
            self.save_model_if_enabled()

        # Save the plots if configured.
        if self.save_plots_automatically == True:
            self.save_plots_if_enabled()


    def save_model_if_enabled(self):
        """
        Saves the model automatically if enabled.
        """

        # Saving the model wrt. frequency or on last iteration.
        if self.current_iteration % self.model_save_frequency == 0 or self.current_iteration == self.number_of_iterations - 1:
            self.save_model("{}-model-{:08d}.h5".format(self.name, self.current_iteration + 1))


    def save_model(self, path):
        """
        Saves the model to a path.

        Args:
            path (string): Where to save the model.
        """

        self.model.save(path)


    def save_plots_if_enabled(self):
        """
        Saves the plots automatically if enabled.
        """
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


    def predict_on_state(self, state):
        """
        Performs one prediction on a state.

        Args:
            state (ndarray): A state of the environment.

        Returns:
            ndarray: A prediction.
        """

        return self.model.predict(np.expand_dims(state, axis=0))[0]


    def get_status_string(self):
        """
        Yields a status string.

        The string contains all data relevant to training.

        Returns:
            string: The status string.
        """

        elapsed_time = time.time() - self.start_time
        estimated_time = self.number_of_iterations * elapsed_time / self.current_iteration - elapsed_time if self.current_iteration != 0 else 0.0

        status_string = ""
        status_string += "{}/{} " .format(self.current_iteration, self.number_of_iterations)
        status_string += "{:.02f}% ".format(100.0 * self.current_iteration / self.number_of_iterations)
        status_string += "elapsed: {} ".format(str(datetime.timedelta(seconds=int(elapsed_time))))
        status_string += "estimated: {} ".format(str(datetime.timedelta(seconds=int(estimated_time))))
        status_string += "epsilon: {:.04f} ".format(self.current_epsilon)
        #status_string += "action: {} ".format(self.current_action)
        #status_string += "random: {} ".format(do_random_action)
        #status_string += "reward: {} ".format(self.current_reward)
        #status_string += "maxq: {:.04f} ".format(self.current_maxq_value)
        if self.track_rewards:
            status_string += "rewards mean: {:.04f} ".format(self.current_rewards_running_means)
        if self.track_episodes:
            status_string += "episodes mean: {:.04f} ".format(self.current_episodes_running_means)
        if self.track_maxq:
            status_string += "maxq mean: {:.04f} ".format(self.current_maxq_running_means)
        status_string += " " * 10
        return status_string


class DDQNAgent(DQNAgent):
    """
    This is the implementation of an DDQN-Agent.

    It allows for doing deep reinforcement learning. It also does
    minibatch-replay against catastrophic forgetting. On top of that it makes
    heavy use of a target net in order to make the solution more stable
    """

    def __init__(self, name, model, number_of_actions, gamma, final_epsilon, initial_epsilon, number_of_iterations, replay_memory_size, minibatch_size, model_copy_interval):
        """
        Creates a DDQN-agent.


        Args:
            See constructor of super-class.

            model_copy_interval (int): Interval of how often to copy the weights.
        """

        super().__init__(name, model, number_of_actions, gamma, final_epsilon, initial_epsilon, number_of_iterations, replay_memory_size, minibatch_size)

        self.model_copy_interval = model_copy_interval
        self.target_model = models.clone_model(self.model)


    def replay_memory_via_minibatch(self):
        """
        See super-class.
        """

        # Sample random minibatch and unpack it.
        minibatch = random.sample(self.replay_memory, min(len(self.replay_memory), self.minibatch_size))
        state_batch = np.array([d[0] for d in minibatch])
        action_batch = np.array([d[1] for d in minibatch])
        reward_batch = np.array([d[2] for d in minibatch])
        state_next_batch = np.array([d[3] for d in minibatch])
        terminal_batch = np.array([d[4] for d in minibatch])

        # Prepare q-values.
        q_values = self.model.predict(state_batch)
        q_values_next = self.model.predict(state_next_batch)
        q_values_next_target = self.target_model.predict(state_next_batch)
        for i in range(len(minibatch)):
            if terminal_batch[i] == False:
                argmax = np.argmax(q_values_next[i])
                q_values[i, np.argmax(action_batch[i])] = reward_batch[i] + self.gamma * q_values_next_target[i][argmax]
            else:
                q_values[i, np.argmax(action_batch[i])] = reward_batch[i]

        # Do gradient descent.
        self.model.train_on_batch(state_batch, q_values)

        # Copy weights to target net.
        if self.number_of_iterations % self.model_copy_interval == 0:
            self.target_model.set_weights(self.model.get_weights())

        # Save the model if configured.
        if self.save_model_automatically == True:
            self.save_model_if_enabled()

        # Save the plots if configured.
        if self.save_plots_automatically == True:
            self.save_plots_if_enabled()


class DQNAgentSpecial(DQNAgent):
    """
    This is the implementation of an DDQN-Agent.

    It allows for doing deep reinforcement learning. It also does
    minibatch-replay against catastrophic forgetting. On top of that it makes
    heavy use of a target net in order to make the solution more stable
    """

    def __init__(self, name, model, gamma, final_epsilon, initial_epsilon, number_of_iterations, replay_memory_size, minibatch_size, model_copy_interval):
        """
        Creates a DDQN-agent.


        Args:
            See constructor of super-class.

            model_copy_interval (int): Interval of how often to copy the weights.
        """

        super().__init__(name=name, model=model, number_of_actions=0, gamma=gamma, final_epsilon=final_epsilon, initial_epsilon=initial_epsilon, number_of_iterations=number_of_iterations, replay_memory_size=replay_memory_size, minibatch_size=minibatch_size)

        self.model_copy_interval = model_copy_interval

        self.target_model = models.clone_model(self.model)

        self.action_sizes = [int(output.shape[1]) for output in model.outputs]


    def get_action(self, state):
        """
        Returns an action.

        This implements epsilon greedy. That is, it returns with a certain probability
        either a random action or a predicted action.

        Args:
            state (ndarray): A state of the environment.

        Returns:
            ndarray: A one-hot-encoded action.
        """

        self.current_epsilon = self.epsilon_decrements[self.current_iteration]

        prediction = self.model.predict(np.expand_dims(state, axis=0))
        self.current_maxq_value = np.max([np.max(p) for p in prediction])

        do_random_action = random.random() <= self.current_epsilon
        if do_random_action:
            actions = []
            for action_size in self.action_sizes:
                action = np.zeros((action_size,))
                action_index = np.random.randint(0, action_size)
                action[action_index] = 1.0
                actions.append(action)
            actions = np.array(actions)
        else:
            actions = []
            for predicted_action in prediction:
                predicted_action = predicted_action[-1]
                action = np.zeros((len(predicted_action),))
                action_index = np.argmax(predicted_action)
                action[action_index] = 1.0
                actions.append(action)
            actions = np.array(actions)

        self.current_action = actions

        for a, size in zip(actions, self.action_sizes):
            assert len(a) == size

        return actions


    def replay_memory_via_minibatch(self):
        """
        See super-class.
        """

        # Sample random minibatch and unpack it.
        minibatch = random.sample(self.replay_memory, min(len(self.replay_memory), self.minibatch_size))
        state_batch = np.array([d[0] for d in minibatch])
        action_batch = np.array([d[1] for d in minibatch])
        reward_batch = np.array([d[2] for d in minibatch])
        state_next_batch = np.array([d[3] for d in minibatch])
        terminal_batch = np.array([d[4] for d in minibatch])

        # Prepare q-values.
        q_values = self.model.predict(state_batch)
        q_values_next = self.model.predict(state_next_batch)
        q_values_next_target = self.target_model.predict(state_next_batch)

        for i in range(len(minibatch)):
            for a in range(len(q_values)):
                if terminal_batch[i] == False:
                    argmax = np.argmax(q_values_next[a][i])
                    q_values[a][i, np.argmax(action_batch[i][a])] = reward_batch[i] + self.gamma * q_values_next_target[a][i][argmax]
                else:
                    q_values[a][i, np.argmax(action_batch[i][a])] = reward_batch[i]

        # Do gradient descent.
        self.model.train_on_batch(state_batch, q_values)

        # Copy weights to target net.
        if self.number_of_iterations % self.model_copy_interval == 0:
            self.target_model.set_weights(self.model.get_weights())

        # Save the model if configured.
        if self.save_model_automatically == True:
            self.save_model_if_enabled()

        # Save the plots if configured.
        if self.save_plots_automatically == True:
            self.save_plots_if_enabled()
