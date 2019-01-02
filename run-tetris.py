import warnings
warnings.filterwarnings("ignore")
import numpy as np
import gym_tetris
import tetrisutils as utils
import sys
import glob
from keras import models
import os

def main():
    print("Loading model...")
    if len(sys.argv) == 1:
        model_paths = glob.glob(os.path.join("pretrained_models", "*.h5"))
        model_paths = [model_path for model_path in model_paths if "tetris" in model_path]
        assert len(model_paths) != 0, "Did not find any models."
        model_paths = sorted(model_paths)
        model_path = model_paths[-1]
    else:
        model_path = sys.argv[1]
    model = models.load_model(model_path)
    print(model_path, "loaded.")

    print("Creating environment...")
    environment = gym_tetris.make('Tetris-v0')

    print("Running ...")
    run(model, environment, verbose="verbose" in sys.argv)


def run(model, environment, verbose):

    observation_absolute_maximums = np.array([2.4, 3.6, 0.27, 3.3])

    # main infinite loop
    iterations = 10
    for iteration in range(iterations):
        print("Iteration:", iteration)

        # Initialize.
        image_data = environment.reset()
        image_data = utils.resize_and_bgr2gray(image_data)
        state = utils.image_data_to_state(image_data)

        terminal = False
        steps = 0
        while terminal == False:

            prediction = model.predict(state.reshape((1, 84, 84, 4)))[0]
            action = np.argmax(prediction)

            # Update state.
            image_data_next, reward, terminal, _ = environment.step(action)
            image_data_next = utils.resize_and_bgr2gray(image_data_next)
            state_next = utils.update_state(state, image_data_next)

            # Render.
            environment.render()

            steps += 1
        print("Run lasted for {} steps.".format(steps))


if __name__ == "__main__":
    main()
