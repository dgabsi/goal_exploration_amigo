
#This code is taken from https://github.com/facebookresearch/adversarially-motivated-intrinsic-goals
#I used tha same Neural netowrk initalization as in AMIGO
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module