import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac_goal_multiple
import numpy as np

#The code is based on https://github.com/lcswillems/rl-starter-files
# Function from https://github.com/lcswillems/rl-starter-files
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

#The code is based on https://github.com/lcswillems/rl-starter-files
class GoalGeneratorModel(nn.Module, torch_ac_goal_multiple.RecurrentACModel):
    def __init__(self, obs_space, device, use_memory=False, use_text=False):
        super().__init__()

        # The code is based on https://github.com/lcswillems/rl-starter-files
        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory
        self.device=device
        self.width = n = obs_space["image"][0]
        self.height = m = obs_space["image"][1]


        # Define image embedding
        # The code is based on https://github.com/lcswillems/rl-starter-files
        self.obs_extract_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ELU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ELU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ELU()
        )
        # The code is based on https://github.com/lcswillems/rl-starter-files
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        # The code is based on https://github.com/lcswillems/rl-starter-files
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        # The code is based on https://github.com/lcswillems/rl-starter-files
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)


        #This part of the code is original
        self.diff_fc = nn.Sequential(
            nn.Linear(m * n, 64),
            nn.ELU(),
            nn.Linear(64, m * n))

        self.initial_fc = nn.Sequential(
            nn.Linear(n*m, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, n*m)
        )

        # The code is based on https://github.com/lcswillems/rl-starter-files
        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        #This part of the code is original
        self.goal_fc = nn.Sequential(
            nn.Linear(self.embedding_size+2*n*m, 64),
            nn.ELU(),
            nn.Linear(64, m * n)
        )
        # Define critic's model
        self.goal_critic = nn.Sequential(
            nn.Linear(n*m, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )

        # The code is based on https://github.com/lcswillems/rl-starter-files
        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory=None, return_distribution=False, init_obs=None, diff=None, carried_col=None, carried_obj=None):

        # this part of the code is original
        x=obs.to(torch.float32).to(self.device)
        #x_init=init_obs.to(self.device)
        B=diff.size()[0]
        x_init = init_obs.to(self.device)

        # this part of the code is original
        diff_hot=torch.zeros((B, self.width*self.height), requires_grad=False, device=self.device)
        for b in range(B):
            if diff[b]>0:
                hot_coded_diff = F.one_hot(diff[b].to(torch.long), num_classes=self.width * self.height)
                hot_coded_diff = hot_coded_diff.reshape(-1, self.width*self.height)
                hot_coded_diff = hot_coded_diff.to(torch.float32).to(self.device)
                diff_hot[b]=hot_coded_diff

        # this part of the code is original
        x_init=x_init.flatten(1,2)
        sum_initial = torch.sum(x_init, -1).to(torch.float32)
        initial_z_score = (sum_initial - torch.mean(sum_initial, dim=-1, keepdims=True)) / torch.std(sum_initial, dim=-1, keepdims=True)

        init_x=self.initial_fc(initial_z_score)

        diff_x=self.diff_fc(diff_hot)

        x = x.permute(0,3,1, 2)#.transpose(2, 3)
        x = self.obs_extract_conv(x)
        x = x.reshape(x.shape[0], -1)

        # The code is based on https://github.com/lcswillems/rl-starter-files
        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        # The code is based on https://github.com/lcswillems/rl-starter-files
        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        #this part of the code is original
        combined_x=torch.cat((embedding,init_x, diff_x), dim=1)

        x= self.goal_fc(combined_x)

        goal_dist = Categorical(logits=x)

        goal=goal_dist.sample()
        goal_log_prob=goal_dist.log_prob(goal)

        goal_value = self.goal_critic(x)
        goal_value = goal_value.squeeze(1)

        if return_distribution:
            return goal, goal_log_prob, goal_value, goal_dist, memory
        else:
            return goal, goal_log_prob, goal_value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]





