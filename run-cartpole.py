import warnings
warnings.filterwarnings("ignore")
import numpy as np
import gym
import sys
import glob
from keras import models


def main():
    print("Loading model...")
    if len(sys.argv) == 1:
        model_paths = glob.glob("*.h5")
        model_paths = [model_path for model_path in model_paths if "cartpole" in model_path]
        model_paths = sorted(model_paths)
        model_path = model_paths[-1]
    else:
        model_path = sys.argv[1]
    model = models.load_model(model_path)
    print(model_path, "loaded.")

    print("Creating environment...")
    environment = gym.make("CartPole-v0")

    print("Running ...")
    run(model, environment, verbose="verbose" in sys.argv)


def run(model, environment, verbose):

    observation_absolute_maximums = np.array([2.4, 3.6, 0.27, 3.3])

    # main infinite loop
    iterations = 10
    for iteration in range(iterations):
        print("Iteration:", iteration)

        # Initialize.
        action = np.array([1.0, 0.0])
        state  = environment.reset()
        state = state / observation_absolute_maximums

        terminal = False
        steps = 0
        while terminal == False:

            prediction = model.predict(state.reshape(1, 4))[0]
            action = np.argmax(prediction)

            # Update state.
            state, reward, terminal, _ = environment.step(action)
            state = state / observation_absolute_maximums
            environment.render()

            steps += 1
        print("Run lasted for {} steps.".format(steps))


if __name__ == "__main__":
    main()
