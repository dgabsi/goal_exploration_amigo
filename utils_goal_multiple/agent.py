import torch

import utils_goal_multiple
from model import ACModel
from goal_generator_model import GoalGeneratorModel
import numpy as np
from amigo_generator_model import  Amigo_GoalGenerator
from amigo_ac_model import Amigo_ACModel

#this is part of torch-ac framework available at https://github.com/lcswillems/torch-ac
#But relevent parts have been changed
class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, obs_space, action_space, model_dir,goal_generator=True,
                 device=None, argmax=False, num_envs=1, use_memory=False, use_text=False, use_amigo=False):
        if not goal_generator:
            with_goal=False
        else: with_goal=True
        obs_space, self.preprocess_obss = utils_goal_multiple.get_obss_preprocessor(obs_space, with_goal)
        if use_amigo:
            # Use amigo type student network
            self.acmodel = Amigo_ACModel(obs_space, action_space, device, hidden_size=256, use_memory=use_memory, use_text=use_text)
        else:
            self.acmodel = ACModel(obs_space, action_space, True, use_memory=use_memory, use_text=use_text)
        self.device = device
        self.argmax = argmax
        self.num_envs = num_envs
        self.obs_space=obs_space
        self.goal = torch.zeros(num_envs)
        self.goal_diff = torch.zeros(num_envs)
        print(obs_space)
        self.goal_obs = torch.zeros(num_envs, *obs_space["image"])


        if self.acmodel.recurrent:
            self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size, device=self.device)

        self.acmodel.load_state_dict(utils_goal_multiple.get_model_state(model_dir))
        self.acmodel.to(self.device)
        self.acmodel.eval()
        if goal_generator:
            if use_amigo:
                self.goal_generator= Amigo_GoalGenerator(obs_space, device, use_memory=use_memory)
            else:
                self.goal_generator= GoalGeneratorModel(obs_space, device=device, use_memory=use_memory)
            self.goal_generator.load_state_dict(utils_goal_multiple.get_goal_generator_model_state(model_dir))
            self.goal_generator.eval()
        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(utils_goal_multiple.get_vocab(model_dir))
        self.goal_memory = torch.zeros(num_envs, self.goal_generator.memory_size)

    def get_actions(self, obss):


        preprocessed_obss = self.preprocess_obss(obss, device=self.device)

        preprocessed_obss.image = preprocessed_obss.image.to(self.device)
        if self.goal_generator:
            goal = preprocessed_obss.goal.to(self.device)
        else:
            goal = None
        with torch.no_grad():
            if self.acmodel.recurrent:
                dist, _, self.memories= self.acmodel(preprocessed_obss, goal, self.memories.to(self.device))
            else:
                dist, _, _ = self.acmodel(preprocessed_obss, goal=goal)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        return actions.cpu().numpy()

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float, device=self.device).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])

    def reset(self):

        self.goal = torch.zeros(self.num_envs)
        self.goal_diff = torch.zeros(self.num_envs)
        self.goal_obs = torch.zeros(self.num_envs, *self.obs_space['image'])
        self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size, device=self.device)
        self.goal_memory = torch.zeros(self.num_envs, self.goal_generator.memory_size)

    def update_goals(self, obs, update=True, argmax=True):
        #Updates goals using the Goal generator(teacher).
        #A goal is updated only if the previous goal is reached ot at the start of episode

        procs_num=len(obs)
        goal_inf_list=[]

        if self.goal_generator is not None:
            self.goal_generator.eval()

            goal=self.goal.clone()
            goal_obs= self.goal_obs.clone()
            goal_diff= self.goal_diff.clone()
            goal_memory=self.goal_memory.clone()

            new_goal=np.zeros(procs_num, dtype=np.float32)
            preprocessed_obs=self.preprocess_obss(obs)
            #teacher_quality_goal=torch.zeros(self.num_procs,requires_grad=False)
            with torch.no_grad():
                if self.goal_generator.recurrent:
                    s_goals, _, _, goal_dist ,s_g_memory = self.goal_generator(preprocessed_obs.image.clone().to(self.device), init_obs=preprocessed_obs.init_image.clone().to(self.device), diff=preprocessed_obs.diff.clone().to(self.device), memory=self.goal_memory.to(self.device) , carried_col=preprocessed_obs.carried_col.clone().to(self.device), carried_obj=preprocessed_obs.carried_obj.clone().to(self.device), return_distribution=True)
                else:
                    s_goals, _, _, goal_dist, s_g_memory = self.goal_generator(preprocessed_obs.image.clone().to(self.device), init_obs=preprocessed_obs.init_image.clone().to(self.device), diff=preprocessed_obs.diff.clone().to(self.device), carried_col=preprocessed_obs.carried_col.clone().to(self.device), carried_obj=preprocessed_obs.carried_obj.clone().to(self.device), return_distribution=True)
                if argmax:
                    s_goals = goal_dist.probs.max(1, keepdim=True)[1]
                else:  s_goals=s_goals.detach().clone().cpu()
                s_g_memory=s_g_memory.detach().clone().cpu()
            for proc in range(procs_num):
                num_step_g=obs[proc]["episode_step"]-obs[proc]["goal_step"]
                #or (preprocessed_obs.episode_step[proc].item()==1)
                #A goal is updated only if the previous goal is reached ot at the start of episode
                goal[proc] = s_goals[proc].squeeze().clone()
                goal_obs[proc] = preprocessed_obs.image[proc].clone()
                goal_diff[proc] = preprocessed_obs.diff[proc].clone()
                new_goal[proc]=1.
                if self.goal_generator.recurrent:
                    goal_memory[proc] = s_g_memory[proc].clone()
                obs[proc]["goal"] = goal[proc].clone().item()
                obs[proc]["goal_image"] = goal_obs[proc].clone().numpy()
                obs[proc]["goal_diff"] = goal_diff[proc].clone().item()
                obs[proc]["goal_step"] = obs[proc]["episode_step"]
                obs[proc]["reached_goal"] = 0
                obs[proc]["last_e_reached"] = 0
                obs[proc]["last_e_step"] = -1
                obs[proc]["last_e_r_weight"] = 0
                    #self.teacher_quality_r[proc] = self.get_cell_potenetial(preprocessed_obs.init_image[proc].clone(), preprocessed_obs.goal[proc].clone())
                goal_inf_list.append({"goal":goal[proc].clone().item(),
                                      "goal_frame":goal_obs[proc].clone().numpy(),
                                      "goal_new": new_goal[proc],
                                      "goal_diff": goal_diff[proc].clone().item()
                                          })
            if update:
                self.goal = goal.clone()
                self.goal_obs = goal_obs.clone()
                self.goal_diff = goal_diff.clone()
                self.goal_memory = goal_memory.clone()

        return obs, goal_inf_list




