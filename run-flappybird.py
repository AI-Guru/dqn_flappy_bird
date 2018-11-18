import warnings
warnings.filterwarnings("ignore")
import numpy as np
from game.flappy_bird import Environment
import sys
import flappybirdutils as utils
import glob
from keras import models


def main():
    print("Loading model...")
    if len(sys.argv) == 1:
        model_paths = glob.glob("*.h5")
        model_paths = [model_path for model_path in model_paths if "flappybird" in model_path]
        model_paths = sorted(model_paths)
        model_path = model_paths[-1]
    else:
        model_path = sys.argv[1]
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
        image_data, reward, terminal = environment.step(action)
        image_data = utils.resize_and_bgr2gray(image_data)
        state = utils.image_data_to_state(image_data)

        while terminal == False:

            prediction = model.predict(state.reshape(1, 84, 84, 4))[0]
            action = np.zeros((2,))
            action[np.argmax(prediction)] = 1.0

            image_data_next, reward, terminal = environment.step(action)

            # Update state.
            image_data_next = utils.resize_and_bgr2gray(image_data_next)
            state = utils.update_state(state, image_data_next)


if __name__ == "__main__":
    main()
