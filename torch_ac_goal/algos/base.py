from abc import ABC, abstractmethod
import torch

from torch_ac_goal.format import default_preprocess_obss
from torch_ac_goal.utils import DictList, ParallelEnv
import numpy as np
from copy import deepcopy
#from obs_utils import analyse_image, analyse_episode, render_image
import matplotlib.pyplot as plt

#the code is and implementation of AMIGo and AMIGo concurrent
#For AMIGo paper:
#Learning with AMIGo: Adversarially Motivated Intrinsic GOals
#(Campero et al., 2015)(https://arxiv.org/abs/2006.12122))

#The implementation is based on the RL torch-ac framework available at https://github.com/lcswillems/torch-ac
#but was changed considerably to add AMIGo capabilities
# as refernce for AMIGo I used  https://github.com/facebookresearch/adversarially-motivated-intrinsic-goals
class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, log_num_return, goal_generator,
                 goal_generator_args, image_model, goal_recurrence, archive_args):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        """

        # Store parameters
        #This is from https://github.com/lcswillems/torch-ac
        self.env = ParallelEnv(envs)
        self.acmodel = acmodel
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward

        #This is new. Goal generator(teacher) data
        self.log_num_return = log_num_return
        self.goal_generator = goal_generator
        self.image_model=image_model
        self.goal_recurrence = goal_recurrence
        self.archive_info = archive_args
        self.goal_generator_info = goal_generator_args
        self.goal_batch_size=goal_generator_args["goal_batch_size"]
        self.archive_info=archive_args

        # Control parameters
        #This is based on https://github.com/lcswillems/torch-ac
        assert self.acmodel.recurrent or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        #This is new
        assert self.goal_generator.recurrent or self.goal_recurrence == 1
        assert self.num_frames_per_proc % self.goal_recurrence == 0

        # Configure acmodel
        # This is based on https://github.com/lcswillems/torch-ac
        self.acmodel=self.acmodel.to(self.device)
        self.acmodel.eval()
        self.goal_generator.eval()

        # Store helpers values

        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Initialize experience values
        # This is based on https://github.com/lcswillems/torch-ac
        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()

        #Student data
        # This is based on https://github.com/lcswillems/torch-ac
        self.obss = [None]*(shape[0])
        if self.acmodel.recurrent:
            self.memory = torch.zeros(shape[1], self.acmodel.memory_size)
            self.memories = torch.zeros(*shape, self.acmodel.memory_size)
        self.mask = torch.ones(shape[1]).to(self.device)
        self.masks = torch.zeros(*shape).to(self.device)
        self.actions = torch.zeros(*shape, dtype=torch.int).to(self.device)
        self.values = torch.zeros(*shape).to(self.device)
        self.rewards = torch.zeros(*shape).to(self.device)
        self.orig_rewards = torch.zeros(*shape).to(self.device)
        self.advantages = torch.zeros(*shape).to(self.device)
        self.log_probs = torch.zeros(*shape).to(self.device)

        #Goal generator data  #This is new
        self.intrin_rewards = torch.zeros(*shape).to(self.device)
        self.init_obss = torch.zeros(*shape, *self.env.observation_space.spaces['image'].shape).to(self.device)
        self.added_goal=False
        self.new_goal = torch.zeros(shape[1]).to(self.device)
        self.goals=torch.zeros(*shape).to(self.device)
        self.goal_values = torch.zeros(*shape).to(self.device)
        self.goal_log_probs = torch.zeros(*shape).to(self.device)
        self.goal=torch.zeros(shape[1]).to(self.device)
        self.goal_value=torch.zeros(shape[1]).to(self.device)
        self.goal_log_prob = torch.zeros(shape[1]).to(self.device)
        self.goal_diff = torch.zeros(shape[1]).to(self.device)
        self.goal_diffs = torch.zeros(*shape).to(self.device)
        self.diff_locs = torch.zeros(*shape).to(self.device)
        self.teacher_rewards = torch.zeros(*shape).to(self.device)
        self.intrin_teacher_rewards = torch.zeros(*shape).to(self.device)
        self.goal_advantages = torch.zeros(*shape).to(self.device)
        self.goal_obss=torch.zeros(*shape, *self.env.observation_space.spaces['image'].shape).to(self.device)
        self.goal_obs = torch.zeros(shape[1], *self.env.observation_space.spaces['image'].shape).to(self.device)
        self.goal_reached_mask = torch.ones(shape[1]).to(self.device)
        self.goal_reached_masks = torch.zeros(*shape).to(self.device)
        if self.goal_generator.recurrent:
            self.goal_memory = torch.zeros(shape[1], self.goal_generator.memory_size).to(self.device)
            self.goal_memories = torch.zeros(*shape, self.goal_generator.memory_size).to(self.device)
        # This is new
        self.exps_goals = None
        self.goal_reached_s=torch.zeros(*shape).to(self.device)
        self.goal_reached_steps_s = torch.zeros(*shape).to(self.device)
        self.goal_reached=torch.zeros(shape[1]).to(self.device)
        self.goal_reached_steps = torch.zeros(shape[1]).to(self.device)

        #Store difficulty
        self.difficulty_counts = torch.zeros(shape[1]).to(self.device)
        self.difficulty = torch.full((shape[1],),self.goal_generator_info["difficulty"]).to(self.device)


        # Initialize log values
        # This is new
        # Logs for goal per proc for current episode
        self.log_goal_count = torch.ones(shape[1]).to(self.device)
        self.log_proc_reach_steps = [[0.] for i in range(shape[1])]
        self.log_proc_reach = [[] for i in range(shape[1])]
        self.log_proc_teacher_return = [[] for i in range(shape[1])]
        #episode logs
        self.goal_episode_reached_count = torch.zeros(shape[1]).to(self.device)
        self.log_episode_goal_count = torch.ones(shape[1]).to(self.device)

        # This is based on https://github.com/lcswillems/torch-ac
        #Episode logs
        self.log_episode_return = torch.zeros(shape[1]).to(self.device)
        self.log_episode_reshaped_return = torch.zeros(shape[1], device=self.device)
        self.log_episode_num_frames = torch.zeros(shape[1]).to(self.device)

        # This is new
        #epsidoe logs for goals
        self.log_episode_goal_reached = torch.zeros(shape[1]).to(self.device)
        self.log_episode_teacher_return = torch.zeros(shape[1]).to(self.device)
        self.log_episode_teacher_intri_return = torch.zeros(shape[1]).to(self.device)
        self.log_episode_reach_steps = torch.zeros(shape[1]).to(self.device)
        self.log_episode_intri_return=torch.zeros(shape[1]).to(self.device)
        #self.goal_episode_reached_count = torch.zeros(shape[1])

        #Logs list# This is based on https://github.com/lcswillems/torch-ac
        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

        # This is new
        self.log_goal_reached = [0] * self.num_procs
        self.log_teacher_return = [0] * self.num_procs
        self.log_reach_steps = [self.difficulty] * self.num_procs
        self.log_goal_count_in_episodes = [1.] * self.num_procs
        self.log_teacher_mean_reward=[0] * self.num_procs
        self.negative_reward_teacher=0.

        #There is an option not to give punishemnt to the agent. Unused for now. For future use.
        # This is new
        self.no_punish=False
        for reshaped_reward in self.reshape_reward:
            if reshaped_reward["type"] == "teacher_goal_reward":
                self.negative_reward_teacher=reshaped_reward["rewards_inf"]["beta"]
                self.no_punish=reshaped_reward["rewards_inf"]["no_punish"]
        self.log_reached_reward=[-self.negative_reward_teacher]* self.num_procs
        self.log_proc_reached_reward = [[-self.negative_reward_teacher] for i in range(shape[1])]

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        next_exps_goals:
            Contains observation, goals, rewards, advantages and other goal attributues . collected for optimization.
            Relevent for Amigo
        """

        #The goal generator and model are in eval state during rollout
        self.goal_generator.eval()
        self.acmodel.eval()

        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            #Update goal information in obs and env # This is new
            self.obs, next_goal_data, _, _, _= self.update_goals(self.obs)
            self.env.update_goal(next_goal_data)


            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)

            # This is based on https://github.com/lcswillems/torch-ac
            #But added with goal information
            #Recieve next action for Student
            with torch.no_grad():
                preporocessed_curr_obs_model=self.preprocess_obss(self.obs, device=self.device)
                preporocessed_curr_obs_model.image=preporocessed_curr_obs_model.image.to(self.device)
                goal=preporocessed_curr_obs_model.goal.to(self.device)
                if self.acmodel.recurrent:
                    dist, value, memory= self.acmodel(preporocessed_curr_obs_model, goal, self.memory.to(self.device) * self.mask.unsqueeze(1).to(self.device))
                else:
                    dist, value, _= self.acmodel(preporocessed_curr_obs_model, goal=goal)
            action = dist.sample()
            dist=dist
            value=value
            memory=memory

            obs, reward, done, _ = self.env.step(action.cpu().numpy())

            # Update experiences values
            # This is based on https://github.com/lcswillems/torch-ac
            self.obss[i] = self.obs
            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, dtype=torch.float, device=self.device)
            self.actions[i] = action
            self.values[i] = value

            ## This is new. collect init observation and diff location
            self.init_obss[i] = preprocessed_obs.init_image.clone()
            self.diff_locs[i] = preprocessed_obs.diff.clone()

            #The goals were updated in the start of current run
            #Collect goal data and add to experiences
            # This is new
            #calculate goal data
            self.goals[i] = self.goal.clone()
            self.goal_values[i] = self.goal_value.clone()
            self.goal_log_probs[i] = self.goal_log_prob.clone()
            self.goal_obss[i] = self.goal_obs.clone()
            self.goal_diffs[i] = self.goal_diff.clone()
            if self.goal_generator.recurrent:
                self.goal_memories[i] = self.goal_memory.clone()

            #this is the new observation. calculate intrinstic reward rewards if exists
            curr_obs = self.preprocess_obss(obs, device=self.device)
            #reward shaping
            # This is new but within the general framework
            #Calculate teacher and student internal rewards
            if len(self.reshape_reward)>0:
                curr_intrin_reward=torch.zeros(self.num_procs).to(self.device)
                curr_teacher_intrin_reward = torch.zeros(self.num_procs).to(self.device)
                for reshaped_reward in self.reshape_reward:
                    #Calculate student intrin reward
                    if (reshaped_reward["type"] == "student_goal_reward"):# This is new .Calculate student reward
                        for proc in range(self.num_procs):
                            episode_step = preprocessed_obs.episode_step[proc].item()
                            max_steps = (self.env.envs[0].max_steps)
                            if not done[proc]:
                                proc_r = reshaped_reward["func"](episode_step, curr_obs.reached_goal[proc].item()>0, max_steps=max_steps, reward_inf=reshaped_reward["rewards_inf"], reached_type=curr_obs.reached_weight[proc].item())
                                curr_intrin_reward[proc] += proc_r
                            elif done[proc]:
                                proc_r = reshaped_reward["func"](episode_step, curr_obs.last_e_reached[proc].item()>0, #curr_obs.last_e_step[proc]
                                                                 max_steps=max_steps, reward_inf=reshaped_reward["rewards_inf"], reached_type=curr_obs.reached_weight[proc].item()) #max_steps=self.env.envs[0].max_steps
                                curr_intrin_reward[proc] += proc_r
                    elif (reshaped_reward["type"] == "teacher_goal_reward"): #Calculate teacher intrin reward# This is new
                        for proc in range(self.num_procs):
                            number_steps_from_g = preprocessed_obs.episode_step[proc].item() - preprocessed_obs.goal_step[proc].item()
                            if not done[proc]:
                                proc_r = reshaped_reward["func"](number_steps_from_g, reshaped_reward["rewards_inf"], self.difficulty[proc], curr_obs.reached_goal[proc].item()>0, curr_obs.diff_type[proc].item())
                                curr_teacher_intrin_reward[proc] += proc_r
                            if done[proc]:
                                proc_r = reshaped_reward["func"](number_steps_from_g,reshaped_reward["rewards_inf"],self.difficulty[proc],  curr_obs.last_e_reached[proc].item()>0, curr_obs.diff_type[proc].item())
                                if ((not curr_obs.last_e_reached[proc].item()>0) and (not reshaped_reward["rewards_inf"]["no_punish"])):
                                    proc_r+=-reshaped_reward["rewards_inf"]["beta"]
                                curr_teacher_intrin_reward[proc] += proc_r
                    else:
                        #Other reward reshaping
                        curr_intrin_reward += torch.tensor([reshaped_reward["func"](obs_, action_, reward_, done_)
                                                              for obs_, action_, reward_, done_ in zip(obs, action, reward, done)])
                #clculate the total for student and teacher rewards
                curr_intrin_reward=torch.clamp(curr_intrin_reward, -1, 1)
                self.rewards[i]= curr_intrin_reward+1.*(torch.tensor(reward, device=self.device)>0).to(torch.float)+torch.tensor(reward, device=self.device)#torch.clamp(torch.tensor(reward)+curr_intrin_reward, -1, 1)
                self.orig_rewards[i]=torch.tensor(reward).to(self.device)
                # This is new .Teacher rewards
                self.intrin_rewards[i]=curr_intrin_reward
                curr_teacher_reshaped_reward = torch.clamp(curr_teacher_intrin_reward, -1, 1)
                self.intrin_teacher_rewards[i] = curr_teacher_reshaped_reward
                self.teacher_rewards[i] = curr_teacher_reshaped_reward*((torch.tensor(reward, device=self.device)<=0).to(torch.float))+torch.tensor(reward, device=self.device)+ 1.*(torch.tensor(reward, device=self.device)>0).to(torch.float)
                #1.*((torch.tensor(reward)>0).to(torch.float))
            else:
                self.rewards[i] = torch.tensor(reward).to(self.device)
                self.orig_rewards[i] = torch.tensor(reward).to(self.device)

            self.log_probs[i] = dist.log_prob(action).to(self.device)


            #Indicate If goal is reached in current step.
            # This is new
            self.goal_reached=((curr_obs.reached_goal+curr_obs.last_e_reached)>=1).to(torch.float)
            self.goal_reached_steps=(preprocessed_obs.episode_step+1 - preprocessed_obs.goal_step)*self.goal_reached
            self.goal_reached_s[i] = self.goal_reached
            self.goal_reached_steps_s[i] = self.goal_reached_steps

            #Goal reached mask
            # This is new
            self.goal_reached_masks[i] = self.goal_reached_mask
            self.goal_reached_mask = 1 - (((curr_obs.reached_goal + (1 - self.mask)) >= 1).to(torch.float))

            # This is based on https://github.com/lcswillems/torch-ac
            self.obs = obs

            # Update log values
            self.log_episode_goal_count += self.new_goal
            # This is based on https://github.com/lcswillems/torch-ac
            self.log_episode_return += torch.tensor(reward, dtype=torch.float).to(self.device)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs).to(self.device)

            # This is new
            self.log_episode_goal_reached+=self.goal_reached_s[i]
            self.log_episode_teacher_return += self.teacher_rewards[i]
            self.log_episode_teacher_intri_return += self.intrin_teacher_rewards[i]
            self.log_episode_reach_steps +=self.goal_reached_steps_s[i]
            self.log_episode_intri_return+=self.intrin_rewards[i]

            for b, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[b].item())

                    # This part is new
                    #Reached count is the number is the number of goals given-1 (unless a teacher goals was reached in the last episode)
                    reached_count=(self.log_episode_goal_reached[b] if curr_obs.last_e_reached[b]>0 else (self.log_episode_goal_reached[b]+1)).item()

                    if (((reached_count>1) and (not curr_obs.last_e_reached[b].item()<=0)) and (self.no_punish)): #future use
                        reached_count-=1 #If the teacher is not punished , do not take the last step into account unless a goals was reached in last
                    if ((self.log_episode_teacher_intri_return[b].item()!=0.) and (reached_count>=1)): #Take into accout also last step
                        episode_intrin_mean_reward=self.log_episode_teacher_intri_return[b].item()/(reached_count)
                        self.log_reached_reward.append(episode_intrin_mean_reward)
                        self.log_proc_reached_reward[b].append(episode_intrin_mean_reward)

                    self.log_reshaped_return.append(self.log_episode_reshaped_return[b].item())
                    self.log_num_frames.append(self.log_episode_num_frames[b].item())
                    # This is new
                    self.log_goal_reached.append((self.log_episode_goal_reached[b].item()/(self.log_episode_goal_count[b].item()-1))*100)
                    self.log_teacher_return.append(self.log_episode_teacher_return[b].item())
                    self.log_teacher_mean_reward.append(self.log_episode_teacher_return[b].item()/reached_count)
                    self.log_proc_teacher_return[b].append(self.log_episode_teacher_intri_return[b].item())
                    if ((self.log_episode_reach_steps[b].item()>0) and (reached_count>=1)):
                        self.log_reach_steps.append(self.log_episode_reach_steps[b].item()/reached_count)
                        self.log_proc_reach_steps[b].append(self.log_episode_reach_steps[b].item() /reached_count)
                    self.log_proc_reach[b].append((self.log_episode_goal_reached[b].item() / (self.log_episode_goal_count[b].item()-1))*100)
                    self.log_goal_count[b]+=self.log_episode_goal_count[b].item()-1
                    self.log_goal_count_in_episodes.append(self.log_episode_goal_count[b].item() - 1)

                    #Update difficulty treshhold
                    #This is new
                    if self.goal_generator_info["with_step_increase"]:
                            difficulty = np.mean(self.log_proc_reach_steps[b][-self.num_procs:]) #difficulty = np.mean(self.log_proc_reach_steps[-self.num_procs:]) (np.mean(self.log_goal_reached[-self.num_procs:]) < self.goal_generator_info["stepi_treshhold"])
                            if (((difficulty <= self.difficulty[b]-1) and (difficulty<self.goal_generator_info["difficulty_max"])) and (np.mean(self.log_proc_reach[b][-self.num_procs:]) > self.goal_generator_info["stepi_treshhold"])):
                                self.difficulty[b]+= 1
                            else:
                                self.difficulty[b] = difficulty+1
                    else:

                            if np.mean(self.log_proc_reached_reward[b][-self.num_procs:]) > self.goal_generator_info["generator_threshold"]:
                                self.difficulty_counts[b] += 1 #np.mean(self.log_teacher_return[-self.num_procs:])
                            else:
                                self.difficulty_counts[b]= 0
                            if self.difficulty_counts[b]>= self.goal_generator_info["difficulty_counts"] and self.difficulty[b]<= self.goal_generator_info["difficulty_max"]:
                                self.difficulty[b]+= 1
                                self.difficulty_counts[b]= 0

                    self.log_episode_goal_count[b] = 1

            # This is based on https://github.com/lcswillems/torch-ac
            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

            #This is new. Reset goals logs for episode
            self.log_episode_goal_reached *= self.mask
            self.log_episode_teacher_return *= self.mask
            self.log_episode_teacher_intri_return *= self.mask
            #self.goal_episode_reached_count*=self.mask
            self.log_episode_reach_steps*=self.mask
            self.log_episode_intri_return*=self.mask

        # This is based on https://github.com/lcswillems/torch-ac
        next_obs = deepcopy(self.obs)
        next_obs, _, _, _, next_goal_values = self.update_goals(next_obs, update=False)
        preporocessed_curr_obs_model = self.preprocess_obss(next_obs)
        preporocessed_curr_obs_model.image = preporocessed_curr_obs_model.image.to(self.device)
        goal = preporocessed_curr_obs_model.goal.to(self.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                _, next_value, _= self.acmodel(preporocessed_curr_obs_model, goal, self.memory.to(self.device)* self.mask.unsqueeze(1).to(self.device))
            else:
                _, next_value, _ = self.acmodel(preporocessed_curr_obs_model, goal=goal)
        #Calculate student advanatages.based on https://github.com/lcswillems/torch-ac
        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        #Calculate teacher advantages
        #this is new
        next_goal_value=next_goal_values
        next_bootstrap_value=next_goal_value
        #This is new . calculate goal advanatages
        if self.goal_generator:#this is new
            for i in reversed(range(self.num_frames_per_proc)):
                next_goal_reached_mask = self.goal_reached_masks[i + 1] if i < self.num_frames_per_proc - 1 else self.goal_reached_mask
                next_mask = self.masks[i + 1] if i < self.num_frames_per_proc - 1 else self.mask
                next_goal_value = self.goal_values[i + 1] if i < self.num_frames_per_proc - 1 else next_goal_value
                next_goal_advantage = self.goal_advantages[i + 1] if i < self.num_frames_per_proc - 1 else 0
                #next_bootstrap_value=next_goal_value*(1-next_goal_reached_mask) +next_goal_reached_mask*next_bootstrap_value
                delta_goal = self.teacher_rewards[i] + 1.* next_goal_value * next_goal_reached_mask - self.goal_values[i]
                self.goal_advantages[i] = delta_goal + 1. * 1. * next_goal_advantage * next_goal_reached_mask



        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        #Prepare batch fro training. flattend the batch rollout after rollout
        # This is based on https://github.com/lcswillems/torch-ac
        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:]).to(self.device)
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1).to(self.device)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1).to(self.device)
        exps.value = self.values.transpose(0, 1).reshape(-1).to(self.device)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1).to(self.device)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1).to(self.device)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1).to(self.device)
        exps.init_obs = self.init_obss.transpose(0, 1).flatten(0,1).to(self.device)

        #Experiences for goals
        #This is new
        if self.goal_generator:
            exps.goal_log_prob = self.goal_log_probs.transpose(0, 1).reshape(-1).to(self.device)
            exps.goal_value = self.goal_values.transpose(0, 1).reshape(-1).to(self.device)
            exps.goal_advantage = self.goal_advantages.transpose(0, 1).reshape(-1).to(self.device)
            exps.goal = self.goals.transpose(0, 1).reshape(-1).to(self.device)
            exps.goal_returnn = exps.goal_value + exps.goal_advantage
            exps.goal_obs = self.goal_obss.transpose(0, 1).reshape(-1, *self.env.observation_space.spaces['image'].shape).to(self.device)
            exps.goal_diff=self.goal_diffs.transpose(0, 1).reshape(-1).to(self.device)
            exps.diff = self.diff_locs.transpose(0, 1).reshape(-1).to(self.device)
            exps.teacher_reward = self.teacher_rewards.transpose(0, 1).reshape(-1).to(self.device)
            if self.goal_generator.recurrent:
                # T x P x D -> P x T x D -> (P * T) x D
                exps.goal_memory = self.goal_memories.transpose(0, 1).reshape(-1, *self.goal_memories.shape[2:]).to(self.device)
                exps.goal_reached_mask = self.goal_reached_masks.transpose(0, 1).reshape(-1).unsqueeze(1).to(self.device)
                # T x P -> P x T -> (P * T) x 1
        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)
        # This is new
        exps.carried_obj=exps.obs.carried_obj.clone()
        exps.carried_col = exps.obs.carried_col.clone()


        batch_exps_goals=None
        # This is new
        #this is for the Amigo implementation. collect only rewarded goals from the experience
        if not self.goal_generator_info["train_together"]:
            exps_goals = DictList()
            reach_ind=(((exps.teacher_reward!=0.).squeeze().to(torch.float))==1.0).squeeze()
            #exps_goals.goal_obs=exps.goal_obs[reach_ind].clone()
            exps_goals.goal_obs = exps.goal_obs[reach_ind]
            exps_goals.goal_value = exps.goal_value[reach_ind]
            exps_goals.goal_advantage= exps.goal_advantage[reach_ind]
            exps_goals.goal = exps.goal[reach_ind].clone()
            exps_goals.goal_returnn = exps.goal_returnn[reach_ind]
            exps_goals.goal_diff = exps.goal_diff[reach_ind]
            exps_goals.goal_log_prob = exps.goal_log_prob[reach_ind]
            exps_goals.goal_memory = exps.goal_memory[reach_ind]
            exps_goals.mask = exps.mask[reach_ind].clone()
            exps_goals.carried_obj = exps.carried_obj[reach_ind].clone()
            exps_goals.carried_col = exps.carried_col[reach_ind].clone()
            exps_goals.init_obs = exps.init_obs[reach_ind].clone()

            batch_exps_goals=self.add_exp_goal(exps_goals)

        # Log some values
        # This is based on https://github.com/lcswillems/torch-ac
        keep_stat=self.log_num_return*self.num_procs
        keep = max(self.log_done_counter, keep_stat)
        # This is based on https://github.com/lcswillems/torch-ac
        #But I added logs for goals
        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "goal_reached_per_episode": self.log_goal_reached[-keep:],
            "teacher_return_per_episode": self.log_teacher_return[-keep:],
            "teacher_mean_reward": self.log_teacher_mean_reward[-keep:],
            "teacher_mean_intrin_reward": self.log_reached_reward[-keep:],
            "difficulty":int(torch.mean(self.difficulty).item()),
            "goal_count": np.mean(self.log_goal_count_in_episodes[-keep:])
        }
        # This is based on https://github.com/lcswillems/torch-ac
        self.log_done_counter = 0
        self.log_return = self.log_return[-keep_stat:]
        self.log_reshaped_return = self.log_reshaped_return[-keep_stat:]
        self.log_num_frames = self.log_num_frames[-keep_stat:]

        #This is new
        self.log_goal_reached = self.log_goal_reached[-keep_stat:]
        self.log_teacher_return = self.log_teacher_return[-keep_stat:]
        self.log_teacher_mean_reward=self.log_teacher_mean_reward[-keep_stat:]
        self.log_reached_reward = self.log_reached_reward[-keep_stat:]

        return exps, logs, batch_exps_goals

    @abstractmethod
    def update_parameters(self):
        pass

    def update_goals(self, obs, update=True):
        #Updates goals using the Goal generator(teacher).
        #A goal is updated only if the previous goal is reached ot at the start of episode
        #This is new

        goal_inf_list=[]

        if self.goal_generator is not None:
            self.goal_generator.eval()

            goal=self.goal.clone()
            goal_log_prob=  self.goal_log_prob.clone()
            goal_value = self.goal_value.clone()
            goal_obs= self.goal_obs.clone()
            goal_diff= self.goal_diff.clone()
            goal_memory=self.goal_memory.clone()

            new_goal=np.zeros(self.num_procs, dtype=np.float32)
            preprocessed_obs=self.preprocess_obss(obs)

            #Generate goals
            with torch.no_grad():
                if self.goal_generator.recurrent:
                    s_goals, s_goal_log_probs, s_goal_values, s_g_memory = self.goal_generator(preprocessed_obs.image.clone().to(self.device), init_obs=preprocessed_obs.init_image.clone().to(self.device), diff=preprocessed_obs.diff.clone().to(self.device), memory=self.goal_memory.to(self.device) * self.goal_reached_mask.unsqueeze(1).to(self.device), carried_col=preprocessed_obs.carried_col.clone().to(self.device), carried_obj=preprocessed_obs.carried_obj.clone().to(self.device))
                else:
                    s_goals, s_goal_log_probs, s_goal_values, _ = self.goal_generator(preprocessed_obs.image.clone().to(self.device), init_obs=preprocessed_obs.init_image.clone().to(self.device), diff=preprocessed_obs.diff.clone().to(self.device), carried_col=preprocessed_obs.carried_col.clone().to(self.device), carried_obj=preprocessed_obs.carried_obj.clone().to(self.device))
                s_goals=s_goals.detach().clone().cpu()
                s_goal_log_probs = s_goal_log_probs.detach().clone().cpu()
                s_goal_values = s_goal_values.detach().clone().cpu()
                s_g_memory=s_g_memory.detach().clone().cpu()
            for proc in range(self.num_procs):
                num_step_g=obs[proc]["episode_step"]-obs[proc]["goal_step"]

                #A goal is updated only if the previous goal is reached ot at the start of episode
                if (((preprocessed_obs.reached_goal[proc].item()>0) or (preprocessed_obs.goal[proc].item() < 0))):
                    goal[proc] = s_goals[proc].squeeze().clone()
                    goal_log_prob[proc]= s_goal_log_probs[proc].squeeze().clone()
                    goal_value[proc] = s_goal_values[proc].squeeze().clone()
                    goal_obs[proc] = preprocessed_obs.image[proc].clone()
                    goal_diff[proc] = preprocessed_obs.diff[proc].clone()
                    new_goal[proc]=1.
                    if self.goal_generator.recurrent:
                        goal_memory[proc] = s_g_memory[proc].clone()
                    #Update the next observation to contain the new goal
                    obs[proc]["goal"] = goal[proc].clone().cpu().item()
                    obs[proc]["goal_image"] = goal_obs[proc].clone().cpu().numpy()
                    obs[proc]["goal_diff"] = goal_diff[proc].clone().cpu().item()
                    obs[proc]["goal_step"] = obs[proc]["episode_step"]
                    obs[proc]["reached_goal"] = 0
                    obs[proc]["last_e_reached"] = 0
                    obs[proc]["last_e_step"] = -1
                    obs[proc]["last_e_r_weight"] = 0
                    self.new_goal[proc]=1

                else:
                    new_goal[proc] = 0
                    self.new_goal[proc] = 0

                goal_inf_list.append({"goal":goal[proc].clone().cpu().item(),
                                      "goal_frame":goal_obs[proc].clone().cpu().numpy(),
                                      "goal_new": new_goal[proc],
                                      "goal_diff": goal_diff[proc].clone().cpu().item()
                                          })
            if update:
                #Update goal information
                self.goal = goal.clone()
                self.goal_log_prob = goal_log_prob.clone()
                self.goal_value = goal_value.clone()
                self.goal_obs = goal_obs.clone()
                self.goal_diff = goal_diff.clone()
                self.goal_memory = goal_memory.clone()

        return obs, goal_inf_list, goal, goal_log_prob, goal_value

    def add_exp_goal(self, curr_exp_goals):
        # This is new
        #Collect goals experiences to be transferred to optimization
        next_exps_goals=None

        #Add to goals buffer the new goal data.
        if self.exps_goals is not None:
            self.exps_goals.goal_obs = torch.cat((self.exps_goals.goal_obs, curr_exp_goals.goal_obs), dim=0)
            self.exps_goals.goal_value = torch.cat((self.exps_goals.goal_value, curr_exp_goals.goal_value), dim=0)
            self.exps_goals.goal_advantage = torch.cat((self.exps_goals.goal_advantage, curr_exp_goals.goal_advantage), dim=0)
            self.exps_goals.goal = torch.cat((self.exps_goals.goal, curr_exp_goals.goal), dim=0)
            self.exps_goals.goal_returnn = torch.cat((self.exps_goals.goal_returnn, curr_exp_goals.goal_returnn), dim=0)
            self.exps_goals.goal_diff = torch.cat((self.exps_goals.goal_diff, curr_exp_goals.goal_diff), dim=0)
            self.exps_goals.goal_log_prob = torch.cat((self.exps_goals.goal_log_prob, curr_exp_goals.goal_log_prob), dim=0)
            self.exps_goals.goal_memory = torch.cat((self.exps_goals.goal_memory, curr_exp_goals.goal_memory), dim=0)
            self.exps_goals.mask = torch.cat((self.exps_goals.mask, curr_exp_goals.mask), dim=0)
            self.exps_goals.carried_obj = torch.cat((self.exps_goals.carried_obj, curr_exp_goals.carried_obj), dim=0)
            self.exps_goals.carried_col = torch.cat((self.exps_goals.carried_col, curr_exp_goals.carried_col), dim=0)
            self.exps_goals.init_obs = torch.cat((self.exps_goals.init_obs, curr_exp_goals.init_obs), dim=0)
        else:
            self.exps_goals=curr_exp_goals

        num_batch=self.goal_batch_size

        if len(self.exps_goals)>=num_batch:
            next_exps_goals = DictList()
            #This is the next batch for optimization

            next_exps_goals.goal_obs = self.exps_goals.goal_obs[:num_batch]
            next_exps_goals.goal_value = self.exps_goals.goal_value[:num_batch]
            next_exps_goals.goal_advantage = self.exps_goals.goal_advantage[:num_batch]
            next_exps_goals.goal = self.exps_goals.goal[:num_batch]
            next_exps_goals.goal_returnn = self.exps_goals.goal_returnn[:num_batch]
            next_exps_goals.goal_diff = self.exps_goals.goal_diff[:num_batch]
            next_exps_goals.goal_log_prob = self.exps_goals.goal_log_prob[:num_batch]
            next_exps_goals.goal_memory = self.exps_goals.goal_memory[:num_batch]
            next_exps_goals.mask = self.exps_goals.mask[:num_batch]
            next_exps_goals.carried_obj = self.exps_goals.carried_obj[:num_batch]
            next_exps_goals.carried_col = self.exps_goals.carried_col[:num_batch]
            next_exps_goals.init_obs = self.exps_goals.init_obs[:num_batch]

            self.exps_goals.goal_obs= self.exps_goals.goal_obs[num_batch:]
            self.exps_goals.goal_value = self.exps_goals.goal_value[num_batch:]
            self.exps_goals.goal_advantage = self.exps_goals.goal_advantage[num_batch:]
            self.exps_goals.goal= self.exps_goals.goal[num_batch:]
            self.exps_goals.goal_returnn = self.exps_goals.goal_returnn[num_batch:]
            self.exps_goals.goal_diff = self.exps_goals.goal_diff[num_batch:]
            self.exps_goals.goal_log_prob = self.exps_goals.goal_log_prob[num_batch:]
            self.exps_goals.goal_memory = self.exps_goals.goal_memory[num_batch:]
            self.exps_goals.mask = self.exps_goals.mask[num_batch:]
            self.exps_goals.carried_obj = self.exps_goals.carried_obj[num_batch:]
            self.exps_goals.carried_col = self.exps_goals.carried_col[num_batch:]
            self.exps_goals.init_obs = self.exps_goals.init_obs[num_batch:]

        return next_exps_goals




