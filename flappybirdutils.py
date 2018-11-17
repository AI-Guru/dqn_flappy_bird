import warnings
warnings.filterwarnings("ignore")
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
import numpy as np
from keras import models, layers, optimizers, initializers


def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (8, 8), strides=(4, 4), activation="relu", input_shape=(84, 84, 4)))
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), activation="relu"))
    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(2, activation="linear"))
    model.compile(
        optimizer=optimizers.Adam(lr=1e-6),
        loss="mse"
    )
    return model


def create_model2():
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


def resize_and_bgr2gray(image):
    image = image[0:288, 0:404]
    image = color.rgb2gray(image)
    image = resize(image, (84, 84), anti_aliasing=True)
    image = np.reshape(image, (84, 84, 1))
    image[image > 0] = 255
    image = image.transpose(2, 0, 1)
    image = image.astype(np.float32) / 255.0 # TODO division right?
    return image


def image_data_to_state(image_data):
    state = np.zeros((84, 84, 4))
    state[:,:,0] = image_data
    state[:,:,1] = image_data
    state[:,:,2] = image_data
    state[:,:,3] = image_data
    return state

def update_state(state, image_data_next):
    state_next = np.zeros((84, 84, 4))
    state_next[:,:,0] = state[:,:,1]
    state_next[:,:,1] = state[:,:,2]
    state_next[:,:,2] = state[:,:,3]
    state_next[:,:,3] = image_data_next
    return state_next


def render_state(state):
    for i in range(state.shape[-1]):
        image_data = state[:,:, i]
        plt.subplot(1, state.shape[-1], i + 1)
        plt.imshow(image_data, cmap="gray")

    plt.show()
    plt.close()
