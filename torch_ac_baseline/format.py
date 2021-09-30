import torch
#########################################################
# This is part of basline framework. code is from https://github.com/lcswillems/rl-starter-files/blob/master/scripts/
#Used to comapre to baseline
###############################################################
def default_preprocess_obss(obss, device=None):
    return torch.tensor(obss, device=device)