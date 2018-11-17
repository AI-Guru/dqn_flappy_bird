from keras import models, layers, optimizers, initializers

class Agent:

    def __init__(self):
        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 100000 # 2000000
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
