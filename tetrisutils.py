import warnings
warnings.filterwarnings("ignore")
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
import numpy as np


def resize_and_bgr2gray(image):
    image = image[17:423, 7:213]
    image = color.rgb2gray(image)
    image = resize(image, (84, 84), anti_aliasing=True)
    image[image > 0] = 255
    image = image.astype(np.float32) / 255.0 # TODO division right?
    return image
