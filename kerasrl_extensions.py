from rl.callbacks import Callback
import tensorflow as tf
from collections import deque
import numpy as np
import time
import datetime

class TensorboardCallback(Callback):
    """
    Provides logging in TensorBoard.
    """

    def __init__(self, path="tensorboard", log_interval=1000, reward_buffer_length=10000, episode_duration_buffer_length=10000):
        self.log_interval = log_interval

        self.iterations = 0

        self.running_data = {}
        self.running_data["reward"] = deque([], maxlen=reward_buffer_length)
        #self.running_data["episode_duration"] = deque([], maxlen=episode_duration_buffer_length)

        self.tensorboard_writer = tf.summary.FileWriter(path, flush_secs=5)

    def on_step_end(self, step, logs={}):
        """
        Logs data if log-interval exceeded.
        """
        self.running_data["reward"].append(logs["reward"])

        if self.iterations % self.log_interval == 0:
            for key, values in self.running_data.items():
                if len(values) > 0:
                    mean = np.mean(values)
                    self._log_scalar(key + "-mean", mean, self.iterations)

        self.iterations += 1

    def on_episode_end(self, step, logs={}):
        """
        Logs the duration of the episode.
        """
        #self.running_data["episode_duration"].append(logs["nb_episode_steps"])

        for key, value in logs.items():
            if key not in self.running_data.keys():
                self.running_data[key] = []
            self.running_data[key].append(logs[key])

    def _log_scalar(self, tag, value, step):
        """
        Accesses tensorbord to log a value.
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.tensorboard_writer.add_summary(summary, step)


class TrainingDurationEstimatorCallback(Callback):
    """
    Provides logging in TensorBoard.
    """

    def __init__(self):
        self.start_time = time.time()

    def on_step_end(self, step, logs={}):

        #elapsed_time = time.time -
        string = str(datetime.timedelta(seconds=666))
