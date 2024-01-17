import wrappers_delay
import dreamer_delay
import tools_delay

import numpy as np
import tensorflow as tf

config = dreamer_delay.define_config()
config.domain = 'pendulum'
config.task = 'swingup'


def make_env(config, datadir, store):
    env = wrappers_delay.DeepMindControl(config.domain, config.task, camera='side')
    env = wrappers_delay.ActionRepeat(env, config.action_count)
    env = wrappers_delay.NormalizeActions(env)
    env = wrappers_delay.TimeLimit(env, config.time_limit / config.action_count)
    if config.delay_step != 0:
        print(config.delay_step)
        env = wrappers_delay.ActionDelay(env, config.delay_step)
    callbacks = []
    if store:
        callbacks.append(lambda ep: tools_delay.save_episodes(datadir, [ep]))
    env = wrappers_delay.Collect(env, callbacks, config.precision)
    env = wrappers_delay.RewardObs(env)
    return env

datadir = config.logdir / 'episodes'
env = make_env(config, datadir, True)
obs = env.reset()
done = None

while not done:
# for i in range(5):
    action = np.ones((1,1))
    action = tf.reshape(action, (1,1))
    action = np.array(action)
    # print(action)
    obs, _, done = env.step(action[0])
    
# file_path = './episodes/20240117-2052-a6743c6e-251.npz'

# data = np.load(file_path)
# for k, v in data.items():
#     print(k)
#     print(v.shape)
#     print(v[0:5])
    
