import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac_goal
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
#But changed to add be add a goal to the observation and other stucture changes
class ACModel(nn.Module, torch_ac_goal.RecurrentACModel):
    def __init__(self, obs_space, action_space, add_goal=True, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory
        self.add_goal=add_goal
        self.in_channels=3

        if add_goal:
            self.in_channels+=1

        # Define image embedding
        #Changed the activation function to be ELU
        self.image_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, (2, 2)),
            #nn.BatchNorm2d(16),
            nn.ELU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            #nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, (2, 2)),
            #nn.BatchNorm2d(64),
            nn.ELU()
        )
        self.width =n= obs_space["image"][0]
        self.height =m= obs_space["image"][1]


        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

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

        # Define actor's model
        #this part is changed from the original version
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 128),
            #nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        # this part is changed from the original version
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 128),
            #nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, goal=None, memory=None):

        image_obs= obs.image

        ##this part is changed from the original version . Adding goal
        if goal is not None:
            hot_coded_goal=F.one_hot(goal.to(torch.long), num_classes=self.width*self.height)
            hot_coded_goal=hot_coded_goal.reshape(-1,self.width, self.height).unsqueeze(-1)
            image_obs=torch.cat((image_obs, hot_coded_goal), dim=-1)
            concat_obs=image_obs.detach().clone()

        # The code is based on https://github.com/lcswillems/rl-starter-files
        x = x = image_obs.permute(0,3,1, 2).to(torch.float32) #image_obs.transpose(1, 3).transpose(2, 3).to(torch.float32)
        #x=obs.image.flatten(-3,-2).transpose(-2,-1)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

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

        #actor network
        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        # Critic network
        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
