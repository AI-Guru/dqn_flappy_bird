import warnings
warnings.filterwarnings("ignore")
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
import numpy as np

def resize_and_bgr2gray(image):
    image = image[0:288, 0:404]
    image = color.rgb2gray(image)
    image = resize(image, (84, 84), anti_aliasing=True)
    image = np.reshape(image, (84, 84, 1))
    image[image > 0] = 255
    image = image.transpose(2, 0, 1)
    image = image.astype(np.float32) / 255.0 # TODO division right?
    return image


def render_state(state):
    for i in range(state.shape[-1]):
        image_data = state[0,:,:, i]
        print(image_data.shape)
        plt.subplot(1, state.shape[-1], i + 1)
        plt.imshow(image_data, cmap="gray")

    plt.show()
    plt.close()
