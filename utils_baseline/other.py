import random
import numpy
import torch
import collections
#This is part of basline framework. code is from https://github.com/lcswillems/rl-starter-files/blob/master/scripts/

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#######
# This is part of basline framework. code is from https://github.com/lcswillems/rl-starter-files/blob/master/scripts/
#Used to comapre to baseline
###############################################################

def seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synthesize(array):
    d = collections.OrderedDict()
    d["mean"] = numpy.mean(array)
    d["std"] = numpy.std(array)
    d["min"] = numpy.amin(array)
    d["max"] = numpy.amax(array)
    return d
