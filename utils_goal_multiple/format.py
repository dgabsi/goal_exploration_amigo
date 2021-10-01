import os
import json
import numpy
import re
import torch
import torch_ac_goal_multiple
import gym

import utils_goal_multiple

#this is part of torch-ac framework available at https://github.com/lcswillems/torch-ac
#But relevent parts have been changed
def get_obss_preprocessor(obs_space, with_goal=True):
    # Check if obs_space is an image space
    if isinstance(obs_space, gym.spaces.Box):
        obs_space = {"image": obs_space.shape}

        def preprocess_obss(obss, device=None):
            return torch_ac_goal_multiple.DictList({
                "image": preprocess_torch(obss, device=device)
            })

    # Check if it is a MiniGrid observation space
    elif isinstance(obs_space, gym.spaces.Dict) and list(obs_space.spaces.keys()) == ["image"]:
        obs_space = {"image": obs_space.spaces["image"].shape, "text": 100}

        vocab = Vocabulary(obs_space["text"])
        def preprocess_obss(obss, device=None):
            if with_goal:
                return torch_ac_goal_multiple.DictList({
                "image": preprocess_torch([obs["image"] for obs in obss], device=device),
                "text": preprocess_texts([obs["mission"] for obs in obss], vocab, device=device),
                "reward": preprocess_torch([obs["reward"] for obs in obss], device=device),
                "done": preprocess_torch([obs["done"] for obs in obss], device=device),
                "episode_return": preprocess_torch([obs["episode_return"] for obs in obss], device=device),
                "episode_step": preprocess_torch([obs["episode_step"] for obs in obss], device=device),
                "episode_win": preprocess_torch([obs["episode_win"] for obs in obss], device=device),
                "diff": preprocess_torch([obs["diff"] for obs in obss], device=device),
                "diff_type": preprocess_torch([obs["diff_type"] for obs in obss], device=device),
                "carried_col": preprocess_torch([obs["carried_col"] for obs in obss], device=device),
                "carried_obj": preprocess_torch([obs["carried_obj"] for obs in obss], device=device),
                "goal": preprocess_torch([obs["goal"] for obs in obss], device=device),
                "reached_goal": preprocess_torch([obs["reached_goal"] for obs in obss], device=device),
                "reached_weight": preprocess_torch([obs["reached_weight"] for obs in obss], device=device),
                "goal_image": preprocess_torch([obs["goal_image"] for obs in obss], device=device),
                 "goal_step": preprocess_torch([obs["goal_step"] for obs in obss], device=device),
                "goal_diff": preprocess_torch([obs["goal_diff"] for obs in obss], device=device),
                "last_e_reached": preprocess_torch([obs["last_e_reached"] for obs in obss], device=device),
                "last_e_step": preprocess_torch([obs["last_e_step"] for obs in obss], device=device),
                "last_e_r_weight": preprocess_torch([obs["last_e_r_weight"] for obs in obss], device=device),
                "init_image": preprocess_torch([obs["init_image"] for obs in obss], device=device)
            })
            else:
                return torch_ac_goal_multiple.DictList({
                    "image": preprocess_torch([obs["image"] for obs in obss], device=device),
                    "text": preprocess_texts([obs["mission"] for obs in obss], vocab, device=device),
                    "reward": preprocess_torch([obs["reward"] for obs in obss], device=device),
                    "done": preprocess_torch([obs["done"] for obs in obss], device=device),
                    "episode_return": preprocess_torch([obs["episode_return"] for obs in obss], device=device),
                    "episode_step": preprocess_torch([obs["episode_step"] for obs in obss], device=device),
                    "episode_win": preprocess_torch([obs["episode_win"] for obs in obss], device=device),
                    "diff": preprocess_torch([obs["diff"] for obs in obss], device=device),
                    "carried_col": preprocess_torch([obs["carried_col"] for obs in obss], device=device),
                    "carried_obj": preprocess_torch([obs["carried_obj"] for obs in obss], device=device)
                })

        preprocess_obss.vocab = vocab

    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss




def preprocess_torch(torch_ent, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    torch_ent = numpy.array(torch_ent)
    return torch.from_numpy(torch_ent).to(device)


def preprocess_texts(texts, vocab, device=None):
    var_indexed_texts = []
    max_text_len = 0

    for text in texts:
        tokens = re.findall("([a-z]+)", text.lower())
        var_indexed_text = numpy.array([vocab[token] for token in tokens])
        var_indexed_texts.append(var_indexed_text)
        max_text_len = max(len(var_indexed_text), max_text_len)

    indexed_texts = numpy.zeros((len(texts), max_text_len))

    for i, indexed_text in enumerate(var_indexed_texts):
        indexed_texts[i, :len(indexed_text)] = indexed_text

    return torch.tensor(indexed_texts, device=device, dtype=torch.long)


class Vocabulary:
    """A mapping from tokens to ids with a capacity of `max_size` words.
    It can be saved in a `vocab.json` file."""

    def __init__(self, max_size):
        self.max_size = max_size
        self.vocab = {}

    def load_vocab(self, vocab):
        self.vocab = vocab

    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]
