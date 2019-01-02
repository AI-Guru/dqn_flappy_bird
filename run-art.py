import warnings
warnings.filterwarnings("ignore")
import numpy as np
import gym
import gym_art
import artutils as utils
import sys
import glob
from keras import models
import os

def main():
    print("Loading model...")
    if len(sys.argv) == 1:
        model_paths = glob.glob(os.path.join("pretrained_models", "*.h5"))
        model_paths = [model_path for model_path in model_paths if "art-model" in model_path]
        assert len(model_paths) != 0, "Did not find any models."
        model_paths = sorted(model_paths)
        model_path = model_paths[-1]
    else:
        model_path = sys.argv[1]
    model = models.load_model(model_path)
    print(model_path, "loaded.")

    print("Creating environment...")
    environment = gym.make("art-mnist-v0")

    print("Running ...")
    run(model, environment, verbose="verbose" in sys.argv)


def run(model, environment, verbose):

    observation_absolute_maximums = np.array([2.4, 3.6, 0.27, 3.3])

    # main infinite loop
    iterations = 10
    for iteration in range(iterations):
        print("Iteration:", iteration)

        # Initialize.
        (observation_target, observation_canvas) = environment.reset()
        state = utils.image_data_to_state(observation_target, observation_canvas, frames=4)

        terminal = False
        steps = 0
        rewards = 0
        while terminal == False:

            # Predict action.
            prediction = model.predict(np.expand_dims(state, axis=0))
            actions = []
            for predicted_action in prediction:
                predicted_action = predicted_action[-1]
                action = np.argmax(predicted_action)
                actions.append(action)
            actions = np.array(actions)

            # Update state.
            (_, observation_canvas_next), reward, terminal, _ = environment.step(actions)
            state = utils.update_state(state, observation_canvas_next)
            rewards += reward

            # Render.
            environment.render()

            print("Step:", steps, "Action:", actions, "Reward:", reward, "Rewards:", rewards, "               ", end="\r")

            steps += 1
        print("")
        print("Run lasted for {} steps.".format(steps))


if __name__ == "__main__":
    main()
