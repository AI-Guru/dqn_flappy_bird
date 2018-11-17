import warnings
warnings.filterwarnings("ignore")

#import random
import numpy as np
#import time
from game.flappy_bird import Environment
import sys
#import matplotlib.pyplot as plt
#from agent import Agent
#import datetime
import utils
import glob
from keras import models

def main():
    print("Loading model...")
    model_paths = sorted(glob.glob("*.h5"))
    model_path = model_paths[-1]
    model = models.load_model(model_path)
    print(model_path, "loaded.")

    print("Creating environment...")
    environment = Environment(headless=False)

    print("Running ...")
    run(model, environment, verbose="verbose" in sys.argv)


def run(model, environment, verbose):

    # main infinite loop
    iterations = 10
    for iteration in range(iterations):

        # Initialize.
        action = np.array([1.0, 0.0])
        image_data, reward, terminal = environment.frame_step(action)
        image_data = utils.resize_and_bgr2gray(image_data)
        state = np.zeros((84, 84, 4))
        state[:,:,0] = image_data
        state[:,:,1] = image_data
        state[:,:,2] = image_data
        state[:,:,3] = image_data

        while terminal == False:

            prediction = model.predict(state.reshape(1, 84, 84, 4))[0]
            action = np.zeros((2,))
            action[np.argmax(prediction)] = 1.0

            image_data_next, reward, terminal = environment.frame_step(action)

            # Update state.
            image_data_next = utils.resize_and_bgr2gray(image_data_next)
            state_next = np.zeros((84, 84, 4))
            state_next[:,:,0] = state[:,:,1]
            state_next[:,:,1] = state[:,:,2]
            state_next[:,:,2] = state[:,:,3]
            state_next[:,:,3] = image_data_next

            state = state_next


if __name__ == "__main__":
    main()
