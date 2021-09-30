import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac_goal_multiple
from amigo_utils import init
import numpy as np


class Amigo_ACModel(nn.Module, torch_ac_goal_multiple.RecurrentACModel):
    def __init__(self, obs_space, action_space, device, hidden_size=256, use_memory=False, use_text=False):
        super().__init__()
        # Neural network for AMIGO student
        # This code is based on https://github.com/facebookresearch/adversarially-motivated-intrinsic-goals
        # and on https://github.com/lcswillems/rl-starter-files

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory
        self.device=device

        self.observation_shape = obs_space
        self.num_actions = action_space.n
        #self.state_embedding_dim = state_embedding_dim

        self.use_index_select = True
        self.obj_dim = 5
        self.col_dim = 3
        self.con_dim = 2
        #self.goal_dim = 10 #info["goal_dim"]
        #self.agent_loc_dim = 10
        self.num_channels = (self.obj_dim + self.col_dim + self.con_dim + 1)

        self.width = n = obs_space["image"][0]
        self.height = m = obs_space["image"][1]

        self.embed_object = nn.Embedding(11, self.obj_dim)
        self.embed_color = nn.Embedding(6, self.col_dim)
        self.embed_contains = nn.Embedding(4, self.con_dim)


        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        if self.width%2>0:
            self.padding_layer=init_(nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=(4, 4), padding=0))
            n-=3
            m-=3

        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(in_channels=self.num_channels, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
        )


        conv_size = lambda w: ((w - 3 + 2) // 2) + 1
        for i in range(4):
            n = conv_size(n)
        for i in range(4):
            m = conv_size(m)

        self.fc_in_size = n * m * 32+self.obj_dim + self.col_dim

        self.fc = nn.Sequential(
            init_(nn.Linear(self.fc_in_size, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )

        self.image_embedding_size=hidden_size

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
                               constant_(x, 0))

        self.actor = init_(nn.Linear(self.embedding_size, action_space.n))
        self.critic = init_(nn.Linear(self.embedding_size, 1))

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, goal=None, memory=None):


        x = obs.image.detach().clone().to(self.device)
        B, w, *_ = x.shape


        # Creating goal_channel
        carried_col = obs.carried_col.detach().clone().to(self.device)
        carried_obj = obs.carried_obj.detach().clone().to(self.device)
        hot_coded_goal=F.one_hot(goal.detach().clone().to(torch.long), num_classes=self.width*self.height)
        hot_coded_goal=hot_coded_goal.reshape(-1,self.width, self.height).unsqueeze(-1).to(self.device)


        x = x.long()
        carried_obj = carried_obj.long()
        carried_col = carried_col.long()

        x = torch.cat([self.create_embeddings(x, 0), self.create_embeddings(x, 1), self.create_embeddings(x, 2),
                           hot_coded_goal.float()], dim=3)
        carried_obj_emb = self._select(self.embed_object, carried_obj)
        carried_col_emb = self._select(self.embed_color, carried_col)


        x =  x.permute(0,3,1, 2).to(torch.float32)

        if self.width % 2 > 0:
            x=self.padding_layer(x)
        x = self.feat_extract(x)
        x = x.view(B, -1)
        carried_obj_emb = carried_obj_emb.view(B, -1)
        carried_col_emb = carried_col_emb.view(B, -1)
        union = torch.cat([x, carried_obj_emb, carried_col_emb], dim=1)
        x = self.fc(union)


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

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

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

