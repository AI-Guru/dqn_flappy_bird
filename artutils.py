import warnings
warnings.filterwarnings("ignore")
from keras import models, layers, optimizers
#from skimage import data, color
#from skimage.transform import rescale, resize, downscale_local_mean
#import matplotlib.pyplot as plt
import numpy as np


def create_model(input_frames, input_dimensions, output_dimensions, cnn_blocks, dense_dimensions):

    model_input = layers.Input(shape=input_dimensions + (1 + input_frames,))

    hidden_layer = model_input
    for _ in range(cnn_blocks):
        hidden_layer = layers.Conv2D(32, (3, 3), activation="relu")(hidden_layer)
        hidden_layer = layers.MaxPooling2D((2, 2))(hidden_layer)

    latent_layer = layers.Flatten()(hidden_layer)
    for dense_dimension in dense_dimensions:
        latent_layer = layers.Dense(dense_dimension, activation="relu")(latent_layer)

    output_layers = []
    for output_dimension in output_dimensions:
        output_layer = layers.Dense(output_dimension, activation="linear")(latent_layer)
        output_layers.append(output_layer)

    model = models.Model(model_input, output_layers)

    model.compile(
        optimizer=optimizers.Adam(lr=1e-6),
        loss="mse"
    )

    return model


def image_data_to_state(observation_target, observation_canvas, frames):
    state = np.zeros(observation_target.shape + (frames + 1,))
    state[:, :, 0] = observation_target
    for i in range(1, frames):
        state[:, :, i] = observation_canvas
    return state


def update_state(state, observation_canvas_next):
    frames = state.shape[-1] - 1
    state_next = np.zeros(state.shape)
    state_next[:, :, 0] = state[:,:,0]
    for i in range(1, frames):
        state_next[:, : ,i] = state[:, :, i + 1]
    state_next[:, :, frames] = observation_canvas_next
    return state_next


def render_state(state):
    for i in range(state.shape[-1]):
        image_data = state[:,:, i]
        plt.subplot(1, state.shape[-1], i + 1)
        plt.imshow(image_data, cmap="gray")

    plt.show()
    plt.close()
