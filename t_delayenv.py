import dreamer
import wrappers
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.mixed_precision import Policy
import pathlib
import os
import shutil

env = wrappers.DeepMindControl('pendulum', 'swingup', 'pendulum', camera='side')
env = wrappers.ActionDelay(env, 14)
env = wrappers.ActionRepeat(env, 2)
env = wrappers.Collect(env, None, precision=16)
env = wrappers.RewardObs(env)

config = dreamer.define_config()
config.batch_size = 2
config.batch_length = 2
config.train_steps = 1
if config.precision == 16:
    set_global_policy(Policy('mixed_float16'))

sourcedir = config.logdir / 'episodes'
datadir = config.logdir / 'episodes_test'
directory = pathlib.Path(datadir).expanduser()
directory.mkdir(parents=True, exist_ok=True)
if not any(file.endswith('.npz') for file in os.listdir(datadir)):
    for filename in sourcedir.glob('*.npz'):
        shutil.copy2(filename, directory)
        break

act = env.action_space
state = env.observation_space
writer = None

agent = dreamer.Dreamer(config, datadir, act, state, writer)
agent.load()

state = None
done = None
obs = env.reset()

while not done:
    action, state = agent.policy(obs, state, training=False)
    action = np.array(action)
    obs, _, done = env.step(action[0])
    print(action[0])
    plt.imshow(obs['image'])
    plt.axis('off')
    plt.show()

