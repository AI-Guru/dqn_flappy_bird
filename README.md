# Deep Reinforcement Learning Playground

Namaste. This is a playground for Deep Reinforcement Learning.

In my quest for mastering the very exciting field of Deep Reinforcement Learning I created this repository. It is my goal to track my progress and make it available to the public.

What you will find in this repository are a couple of ready-made Deep Reinforcement Learning agents. Including [DQN agent](https://deepmind.com/research/dqn/) and [Double DQN agent](https://arxiv.org/abs/1509.06461). Also you will find training-scripts that apply the agent on multiple environments. Most of those are compatible to [OpenAI Gym](https://gym.openai.com).


## Running.

- `python run-cartpole.py pretrained_models/cartpole-model.h5`
- `python run-flappybird.py pretrained_models/flappybird-model.h5`

## Training.

- `python train-cartpole.py headless`
- `python train-flappybird.py headless`

Note: The `headless` parameter suppresses the renderer of the respective environment. This makes training perfectly suitable for running on a window-server-less server. Omit the parameter if you would like to watch the agent interact with tehe environment during training. This will definitely extend the duration of training.

## Cudos.

- [https://github.com/nevenp/dqn_flappy_bird](https://github.com/nevenp/dqn_flappy_bird)
