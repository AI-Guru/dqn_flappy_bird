
import sys
sys.path.append("/Users/tristanbehrens/Development/DeepLearning/thirdparty/donkey_rl/donkey_rl/src/donkey_gym")
import gym
import donkey_gym
import matplotlib.pyplot as plt
from skimage import io

environment = gym.make("donkey-generated-roads-v0")
observation = environment.reset()

print(observation.shape)

for i in range(100):
    print(i)
    observation, reward, _, _ = environment.step([0.1, 0.1])
    io.imsave("{}-{}.png".format(i, reward), observation)
