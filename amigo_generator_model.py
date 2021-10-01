import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac_goal_multiple
from amigo_utils import init
import numpy as np


#This code is based on https://github.com/facebookresearch/adversarially-motivated-intrinsic-goals
#and on https://github.com/lcswillems/rl-starter-files
#But changed to add diff location and other stucture changes
class Amigo_GoalGenerator(nn.Module, torch_ac_goal_multiple.RecurrentACModel):
    def __init__(self, obs_space, device, use_memory=False, use_text=False, hidden_size=256):
        super().__init__()
        #Neural network for AMIGO TEACHER (Goal generator)
        # This code is based on https://github.com/facebookresearch/adversarially-motivated-intrinsic-goals
        # and on https://github.com/lcswillems/rl-starter-files

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory
        self.device=device


        self.observation_shape = obs_space

        self.use_index_select = True
        self.obj_dim = 5
        self.col_dim = 3
        self.con_dim = 2

        self.num_channels = (self.obj_dim + self.col_dim + self.con_dim+1)

        self.width = n = obs_space["image"][0]
        self.height = m = obs_space["image"][1]
        self.env_dim = self.width * self.height

        self.embed_object = nn.Embedding(11, self.obj_dim)
        self.embed_color = nn.Embedding(6, self.col_dim)
        self.embed_contains = nn.Embedding(4, self.con_dim)


        K = self.num_channels  # number of input filters
        F = 3  # filter dimensions
        S = 1  # stride
        P = 1  # padding
        M = 16  # number of intermediate filters
        Y = 8  # number of output filters
        L = 4  # number of convnet layers
        E = 1  # output of last layer

        in_channels = [K] + [M] * 4
        out_channels = [M] * 3 + [E]

        conv_extract = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]

        def interleave(xs, ys):
            return [val for pair in zip(xs, ys) for val in pair]

        self.extract_representation = nn.Sequential(
            *interleave(conv_extract, [nn.ELU()] * len(conv_extract))
        )

        self.out_dim = self.env_dim * 16 + self.obj_dim + self.col_dim


        self.image_embedding_size=self.env_dim #hidden_size

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size


        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        self.fc = nn.Sequential(
            init_(nn.Linear(self.env_dim, self.env_dim)),
            nn.ELU(),
            init_(nn.Linear(self.env_dim, self.env_dim)),
            nn.ELU(),
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        self.baseline_teacher = init_(nn.Linear(self.env_dim, 1))

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory=None, return_distribution=False, init_obs=None, diff=None, carried_col=None, carried_obj=None):


        x = obs.to(torch.float32).to(self.device)

        B = diff.size()[0]


        # Adding the diff location to the observation
        diff_hot = torch.zeros((B, self.width, self.height), requires_grad=False, device=self.device)
        for b in range(B):
            if diff[b] > 0:
                hot_coded_diff = F.one_hot(diff[b].to(torch.long), num_classes=self.width * self.height)
                hot_coded_diff = hot_coded_diff.reshape(-1, self.width, self.height)
                hot_coded_diff = hot_coded_diff.to(torch.float32).to(self.device)
                diff_hot[b] = hot_coded_diff

        #Not used
        carried_col = carried_col.detach().clone().to(self.device)
        carried_obj = carried_obj.detach().clone().to(self.device)


        x = x.long()
        carried_obj = carried_obj.long()
        carried_col = carried_col.long()

        x = torch.cat([self.create_embeddings(x, 0), self.create_embeddings(x, 1), self.create_embeddings(x, 2), diff_hot.unsqueeze(-1).float()], dim=3)

        x =  x.permute(0,3,1, 2).to(torch.float32)
        x = self.extract_representation(x)
        x = x.view(B, -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        embedding=embedding.view(B, -1)

        embedding = self.fc(embedding)

        generator_logits = embedding.view(B, -1)

        #The goal critic netowrk
        goal_value = self.baseline_teacher(generator_logits)

        #Goal distribution to generate goal (actor in the goal generator context)
        goal_dist = Categorical(logits=generator_logits)

        goal = goal_dist.sample()
        goal_log_prob = goal_dist.log_prob(goal)


        if return_distribution:
            return goal, goal_log_prob, goal_value, goal_dist, memory
        else:
            return goal, goal_log_prob, goal_value, memory


    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]

    def initial_state(self, batch_size):
        """Initializes LSTM."""
        if not self.use_lstm:
            return tuple()
        return tuple(torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size) for _ in range(2))

    def create_embeddings(self, x, id):
        """Generates compositional embeddings."""
        if id == 0:
            objects_emb = self._select(self.embed_object, x[:, :, :, id::3])
        elif id == 1:
            objects_emb = self._select(self.embed_color, x[:, :, :, id::3])
        elif id == 2:
            objects_emb = self._select(self.embed_contains, x[:, :, :, id::3])
        embeddings = torch.flatten(objects_emb, 3, 4)
        return embeddings

    def _select(self, embed, x):
        """Efficient function to get embedding from an index."""
        if self.use_index_select:
            out = embed.weight.index_select(0, x.reshape(-1))
            # handle reshaping x to 1-d and output back to N-d
            return out.reshape(x.shape + (-1,))
        else:
            return embed(x)


    def get_critic_value(self, goal_logits):

        return self.baseline_teacher(goal_logits)