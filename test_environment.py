from game.flappy_bird import Environment
import numpy as np
import utils

environment = Environment(headless=True)

# main infinite loop
iterations = 10
for iteration in range(iterations):
    print("Iteration", iteration)

    # Initialize.
    action = np.array([1.0, 0.0])

    terminal = False
    while terminal == False:

        #utils.render_state(state)
        #print(np.min(state), np.max(state))

        # Do action.
        image_data_next, reward, terminal = environment.frame_step(action)

        # Update state.
        #image_data_next = utils.resize_and_bgr2gray(image_data_next)
        #state = utils.update_state(state, image_data_next)
