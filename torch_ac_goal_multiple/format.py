import torch
#this is part of torch-ac framework available at https://github.com/lcswillems/torch-ac
def default_preprocess_obss(obss, device=None):
    return torch.tensor(obss, device=device)