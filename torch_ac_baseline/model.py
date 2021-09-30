from abc import abstractmethod, abstractproperty
import torch.nn as nn
import torch.nn.functional as F
#########################################################
# This is part of basline framework. code is from https://github.com/lcswillems/rl-starter-files/blob/master/scripts/
#Used to comapre to baseline
###############################################################
class ACModel:
    recurrent = False

    @abstractmethod
    def __init__(self, obs_space, action_space):
        pass

    @abstractmethod
    def forward(self, obs):
        pass

class RecurrentACModel(ACModel):
    recurrent = True

    @abstractmethod
    def forward(self, obs, memory):
        pass

    @property
    @abstractmethod
    def memory_size(self):
        pass