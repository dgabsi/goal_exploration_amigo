import gym
import gym_minigrid

#######
# This is part of basline framework. code is from https://github.com/lcswillems/rl-starter-files/blob/master/scripts/
#Used to comapre to baseline
###############################################################
def make_env(env_key, seed=None, fix_seed=True):
    #Creates a fully observable envvironemnt , with the the dedicated observation wrapper that calculate diff and ass
    env = gym.make(env_key)
    if fix_seed:
        env.seed(seed)
    return env
