from __future__ import print_function

import gym
from gym import logger
import numpy as np
import argparse
import time
import gym.envs.atari as atari
import gym.envs.mujoco as mujoco


def random_rollout(env, num_steps):
    states = []

    for _ in range(num_steps):
        random_action = env.action_space.sample()
        (ob, reward, done, _info) = env.step(random_action)

        env.render()
        time.sleep(1)

        state = env.get_state_()
        states.append(state)
        print("Getting state", state)

    return states

# In default Pong environment, checkpoint a state, then rollout for several more steps.
# Then, reset to the checkpoint. Afterwards, the rendered environment is visibly reset.
def main():
    logger.set_level(logger.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--target', nargs="?", default="atari")
    args = parser.parse_args()

    #env = gym.make(args.target)
    target = args.target
    if target == "atari":
        env = atari.AtariEnv()
    if target == "mujoco":
        env = mujoco.MujocoEnv()

    env.reset()
    num_steps = 8

    # Do a short random rollout
    states = random_rollout(env, num_steps)

    # Restore state to near the beginning of the random walk
    benchmark = states[1]
    print("Setting state", benchmark)
    env.set_state_(benchmark)

    # Observe that the state was reset to the saved benchmark
    random_rollout(env, num_steps)

    env.close()

if __name__ == '__main__':
    main()
