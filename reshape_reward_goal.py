
#this part of the code is original. But based on https://github.com/facebookresearch/adversarially-motivated-intrinsic-goals
#since it is replication the Amigo reward

def student_goal_reward(eps_step, reached=False, max_steps=1, reward_inf=None, reached_type=None):

    #Student intrin reward calculation. based on the number of steps till reached
    reward = reached * ((1 - 0.9 * (float(eps_step) / max_steps)))
    return reward

def teacher_goal_reward(eps_step, reward_inf, difficulty, reached=False, reached_type=None):
    # Teacher intrin reward calculation. if reached after more then difficulty steps, a positive reward is given else negative reward
    reward=0
    if reached:
        if eps_step>=difficulty:
            reward=reward_inf["alpha"]
        if eps_step<difficulty:
            reward=-reward_inf["beta"]
    return reward


