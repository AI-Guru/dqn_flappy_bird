from keras import models, layers, optimizers, initializers


def create_model(number_of_actions):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (8, 8), strides=(4, 4), activation="relu", input_shape=(84, 84, 4)))
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), activation="relu"))
    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(number_of_actions, activation="linear"))
    model.compile(
        optimizer=optimizers.Adam(lr=1e-6),
        loss="mse"
    )
    return model

def create_model2(number_of_actions):
    kernel_initializer = initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=None)
    bias_initializer = initializers.Constant(value=0.01)

    model = models.Sequential()
    model.add(layers.Conv2D(32, (8, 8), strides=(4, 4), activation="relu", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, input_shape=(84, 84, 4)))
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), activation="relu", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation="relu", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
    model.add(layers.Dense(number_of_actions, activation="linear", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
    model.compile(
        optimizer=optimizers.Adam(lr=1e-6),
        loss="mse"
    )
    return model
