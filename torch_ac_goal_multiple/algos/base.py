from abc import ABC, abstractmethod
import torch

from torch_ac_goal_multiple.format import default_preprocess_obss
from torch_ac_goal_multiple.utils import DictList, ParallelEnv
import numpy as np
from copy import deepcopy
# from obs_utils import analyse_image, analyse_episode, render_image
import matplotlib.pyplot as plt
import cv2
from obs_utils import reset_obs


#the code is and implementation of AMIGo Multiple
#For AMIGo paper:
#Learning with AMIGo: Adversarially Motivated Intrinsic GOals
#(Campero et al., 2015)(https://arxiv.org/abs/2006.12122))

#The implementation is based on the RL torch-ac framework available at https://github.com/lcswillems/torch-ac
#but was changed considerably to add AMIGo capabilities
# as refernce for AMIGo I used  https://github.com/facebookresearch/adversarially-motivated-intrinsic-goals
class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, log_num_return,
                 goal_generator,
                 goal_generator_args, image_model, goal_recurrence, archive_args):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the student model
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
        reshape_reward : list of reshaped return functions and args
            list of a functions that shapes the reward
        log_num_return:
            number of history for running mean statistics
        goal_generator:
            teacher network
        goal_generator_args:
            teacher network args
        goal_recurrence: int
            the number of steps the goal gradient is propagated back in time

        """
        #This is based on https://github.com/lcswillems/torch-ac
        # Store parameters
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

        # This part is new. adding the goal generator(teacher) and other goal attibutes
        self.log_num_return = log_num_return
        self.goal_generator = goal_generator
        self.image_model = image_model
        self.goal_recurrence = goal_recurrence
        self.archive_info = archive_args
        self.goal_generator_info = goal_generator_args
        self.goal_batch_size = goal_generator_args["goal_batch_size"]

        # Control parameters
        assert self.acmodel.recurrent or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # This part is new. adding the goal generator(teacher) and other goal attibutes
        assert self.goal_generator.recurrent or self.goal_recurrence == 1
        assert self.num_frames_per_proc % self.goal_recurrence == 0


        # #This is based on https://github.com/lcswillems/torch-ac
        self.acmodel = self.acmodel.to(self.device)
        self.acmodel.eval()
        self.goal_generator.eval()

        # Store helpers values
        # this part of the code is based on the torch-ac framework available at https://github.com/lcswillems/torch-ac
        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Initialize experience values
        # This is based on https://github.com/lcswillems/torch-ac
        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        # batch experience for student
        self.obss = [None] * (shape[0])
        if self.acmodel.recurrent:
            self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, dtype=torch.int, device=self.device)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.orig_rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        # This part is new. adding the placeholders for goals values
        self.intrin_rewards = torch.zeros(*shape, device=self.device)
        self.init_obss = torch.zeros(*shape, *self.env.observation_space.spaces['image'].shape, device=self.device)
        self.eps_step = torch.zeros(*shape, device=self.device)
        self.student_goal = torch.zeros(*shape, device=self.device)

        # batch experience for teacher
        # This part is new. adding the placeholders for goals values
        #We holds goals for the entire episode
        self.added_goal = False
        self.goal_num_frames = self.num_frames_per_proc * self.num_procs
        eps_shape = (self.num_procs, self.env.envs[0].max_steps + 1)
        self.goals = torch.full(eps_shape, -1, device=self.device)
        self.goal_steps = torch.zeros(*eps_shape, device=self.device)
        self.goal_step = torch.zeros(*eps_shape, device=self.device)
        self.goal_values = torch.zeros(*eps_shape, device=self.device)
        self.goal_log_probs = torch.zeros(*eps_shape, device=self.device)
        self.goal_rewards = torch.zeros(*eps_shape, device=self.device)
        self.goal = torch.zeros(shape[1], device=self.device)
        self.goal_value = torch.zeros(shape[1], device=self.device)
        self.goal_log_prob = torch.zeros(shape[1], device=self.device)
        self.goal_diff = torch.zeros(shape[1], device=self.device)
        self.goal_diffs = torch.zeros(*eps_shape, device=self.device)
        self.diff_locs = torch.zeros(*shape, device=self.device)
        self.teacher_rewards = torch.zeros(*eps_shape, device=self.device)
        self.intrin_teacher_rewards = torch.zeros(*eps_shape, device=self.device)
        self.goal_advantages = torch.zeros(*eps_shape, device=self.device)
        self.goal_obss = torch.zeros(*eps_shape, *self.env.observation_space.spaces['image'].shape, device=self.device)
        self.goal_obs = torch.zeros(shape[1], *self.env.observation_space.spaces['image'].shape, device=self.device)
        self.goal_reached = torch.zeros(shape[1], device=self.device)
        self.goals_reached = torch.zeros(*eps_shape, device=self.device)
        self.goal_masks = torch.ones(*eps_shape, device=self.device)
        self.goal_reached_masks = torch.ones(*eps_shape, device=self.device)
        self.new_goal = torch.zeros(shape[1], device=self.device)
        if self.goal_generator.recurrent:
            self.goal_memory = torch.zeros(shape[1], self.goal_generator.memory_size, device=self.device)
            self.goal_memories = torch.zeros(*eps_shape, self.goal_generator.memory_size, device=self.device)
        self.teacher_quality_r = torch.zeros(shape[1], device=self.device)
        self.reach_steps = torch.zeros(*eps_shape, device=self.device)
        self.goal_curr_ind = torch.zeros(shape[1], device=self.device).to(torch.long)
        self.goal_returns = torch.zeros(*eps_shape, device=self.device)
        self.goals_carried_obj = torch.zeros(*eps_shape, device=self.device)
        self.goals_carried_col = torch.zeros(*eps_shape, device=self.device)
        self.goals_init_obs = torch.zeros(*eps_shape, *self.env.observation_space.spaces['image'].shape,
                                          device=self.device)

        # This part is new. adding the difficulty values and goal reached and reached steps indictors
        self.difficulty_counts = torch.zeros(shape[1], device=self.device)
        self.difficulty = torch.full((shape[1],), self.goal_generator_info["difficulty"], device=self.device)
        self.episode_visited = [DictList() for proc in range(self.num_procs)]
        self.exps_goals = None
        self.goal_reached_s = torch.zeros(*shape, device=self.device)
        self.goal_reached_steps_s = torch.zeros(*shape, device=self.device)
        self.goal_reached = torch.zeros(shape[1], device=self.device)
        self.goal_reached_steps = torch.zeros(shape[1], device=self.device)
        self.change_goal = torch.zeros(shape[1], device=self.device)
        self.iter = 0

        # This part is new. adding the placeholders for goals logs
        # Initialize log values
        # Logs for goal per proc in all episodes.totals
        self.log_goal_count = torch.ones(shape[1])
        self.log_proc_reach_steps = [[] for i in range(shape[1])]
        self.log_proc_reach = [[] for i in range(shape[1])]
        self.log_proc_teacher_reward = [[] for i in range(shape[1])]

        # Episode logs
        # This is based on https://github.com/lcswillems/torch-ac
        self.goal_episode_reached_count = torch.zeros(shape[1], device=self.device)
        self.log_episode_goal_count = torch.ones(shape[1], device=self.device)
        self.log_episode_return = torch.zeros(shape[1], device=self.device)
        self.log_episode_reshaped_return = torch.zeros(shape[1], device=self.device)
        self.log_episode_num_frames = torch.zeros(shape[1], device=self.device)

        # This part is new. adding logs for goals values
        self.log_episode_goal_reached = torch.zeros(shape[1], device=self.device)
        # self.log_episode_teacher_return = torch.zeros(shape[1])
        # self.log_episode_teacher_intri_return = torch.zeros(shape[1])
        self.log_episode_reach_steps = torch.zeros(shape[1], device=self.device)

        # Logs lists for all episodes
        # This is based on https://github.com/lcswillems/torch-ac
        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

        # This part is new. adding logs for goals
        self.log_goal_reached = [0] * self.num_procs
        self.log_teacher_reward = [0] * self.num_procs
        self.log_reach_steps = [0] * self.num_procs
        self.log_intrin_teacher_reward = [0.0] * self.num_procs
        self.log_goal_count_in_episodes = [1.] * self.num_procs
        self.log_distinct_goals_epis = [0] * self.num_procs

        self.next_exps_goals = None

    def collect_experiences(self):
        """Collects rollouts and computes advantages.
        #based on the torch-ac framework available at https://github.com/lcswillems/torch-ac
        #but changed considerably to add AMIGo capabilities which for reference I used https://github.com/facebookresearch/adversarially-motivated-intrinsic-goals

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
            Collected in full episodes length

        """
        # Not training models during rollout
        self.goal_generator.eval()
        self.acmodel.eval()
        self.iter += 1
        self.next_exps_goals = None

        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction
            # Update goal information in obs and env
            # This is new
            self.obs, next_goal_data, _, _, _ = self.update_goals(self.obs)
            self.env.update_goal(next_goal_data)

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)

            # This is new. Save init adn diff locaation
            self.init_obss[i] = preprocessed_obs.init_image.clone()
            self.diff_locs[i] = preprocessed_obs.diff.clone()

            # get value and action distribution form student network

            # only adding the goal is new
            # This is based on https://github.com/lcswillems/torch-ac
            with torch.no_grad():
                preporocessed_curr_obs_model = self.preprocess_obss(self.obs, device=self.device)
                preporocessed_curr_obs_model.image = preporocessed_curr_obs_model.image.to(self.device)
                goal = self.goal.to(self.device)
                if self.acmodel.recurrent:
                    dist, value, memory = self.acmodel(preporocessed_curr_obs_model, goal,
                                                       self.memory.to(self.device) * self.mask.unsqueeze(1).to(
                                                           self.device))
                else:
                    dist, value, _ = self.acmodel(preporocessed_curr_obs_model, goal=goal)

            action = dist.sample()

            dist = dist
            value = value
            memory = memory

            obs, reward, done, _ = self.env.step(action.cpu().numpy())

            # Update student experiences values
            # This is based on https://github.com/lcswillems/torch-ac
            self.obss[i] = self.obs
            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, dtype=torch.float, device=self.device)
            self.actions[i] = action
            self.values[i] = value
            self.log_probs[i] = dist.log_prob(action)
            self.student_goal[i] = self.goal

            ## Update teacher experiences values
            # The goals were updated in the start of current run in the update_goals function
            # this is the new observation. used to extract rewards if exists (The new observation attributes will inidctae if the goal is reached)
            curr_obs = self.preprocess_obss(obs, device=self.device)

            # reward shaping
            # All of This part is new but within the framework of https://github.com/lcswillems/torch-ac
            if len(self.reshape_reward) > 0:
                curr_intrin_reward = torch.zeros(self.num_procs).to(self.device)
                for reshaped_reward in self.reshape_reward:
                    # student reshaped reward
                    if (reshaped_reward["type"] == "student_goal_reward"):
                        # student reshaped reward
                        for proc in range(self.num_procs):
                            number_steps_from_g = preprocessed_obs.episode_step[proc].item() + 1
                            max_steps = (self.env.envs[0].max_steps)
                            if not done[proc]:
                                proc_r = reshaped_reward["func"](number_steps_from_g,
                                                                 curr_obs.reached_goal[proc].item() > 0,
                                                                 max_steps=max_steps,
                                                                 reward_inf=reshaped_reward["rewards_inf"],
                                                                 reached_type=curr_obs.reached_weight[proc].item())
                                curr_intrin_reward[proc] += proc_r
                            elif done[proc]: #Check maybe the last step of the episode ended with goal reaching
                                proc_r = reshaped_reward["func"](number_steps_from_g,
                                                                 curr_obs.last_e_reached[proc].item() > 0,
                                                                 max_steps=max_steps,
                                                                 reward_inf=reshaped_reward["rewards_inf"],
                                                                 reached_type=curr_obs.reached_weight[
                                                                     proc].item())  # max_steps=self.env.envs[0].max_steps
                                curr_intrin_reward[proc] += proc_r
                curr_intrin_reward = torch.clamp(curr_intrin_reward, -1, 1)
                # Total student reward. clamp the reward between -1 and 1.  Any  extrnal reward is given is changed to the max value of 1.
                self.rewards[i] = curr_intrin_reward + (
                            1. * (torch.tensor(reward, device=self.device) > 0).to(torch.float)) + torch.tensor(reward,
                                                                                                                device=self.device)
                self.orig_rewards[i] = torch.tensor(reward, device=self.device)
                self.intrin_rewards[i] = curr_intrin_reward #ony the intrinsic reward
                # Update teacher rewards backward if goal was reached
                self.update_hinsight_reward_teacher(torch.tensor(reward, device=self.device), preprocessed_obs,
                                                    curr_obs, done)
            else:# This is based on https://github.com/lcswillems/torch-ac
                self.rewards[i] = torch.tensor(reward, device=self.device)
                self.orig_rewards[i] = torch.tensor(reward, device=self.device)

            # real goal reached mask . 1 if goal reached
            # this part of the code is new
            self.goal_reached = ((curr_obs.reached_goal + curr_obs.last_e_reached) >= 1).to(torch.float).to(self.device)
            self.goal_reached_steps = (
                                                  preprocessed_obs.episode_step + 1 - preprocessed_obs.goal_step) * self.goal_reached
            self.goal_reached_s[i] = self.goal_reached
            self.goal_reached_steps_s[i] = self.goal_reached_steps
            self.eps_step[i] = preprocessed_obs.episode_step

            # Change running observation to the new obs for next step
            # based on framework of https://github.com/lcswillems/torch-ac
            self.obs = obs

            # Update log values
            self.log_episode_goal_count += torch.ones(self.num_procs, device=self.device)
            self.log_episode_return += torch.tensor(reward, dtype=torch.float, device=self.device)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)
            self.log_episode_goal_reached += self.goal_reached
            self.log_episode_reach_steps += self.goal_reached_steps

            # based on framework of https://github.com/lcswillems/torch-ac
            #But changed to add goal logs
            #Calilcating logs for gaols values is new
            for b, done_ in enumerate(done):
                if done_:  # Update logs at the end of episode
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[b].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[b].item())
                    self.log_num_frames.append(self.log_episode_num_frames[b].item())
                    # this is part of the code is new. calculate logs for goal values
                    self.log_goal_reached.append(
                        (self.log_episode_goal_reached[b].item() / (self.log_episode_goal_count[b].item() - 1)) * 100)
                    if self.log_episode_reach_steps[b].item() != 0:
                        self.log_reach_steps.append(
                            self.log_episode_reach_steps[b].item() / (self.log_episode_goal_count[b].item() - 1))
                        self.log_proc_reach_steps[b].append(
                            self.log_episode_reach_steps[b].item() / (self.log_episode_goal_count[b].item() - 1))
                    # this is part of the code is new. calculate logs for goal values
                    self.log_proc_reach[b].append(
                        (self.log_episode_goal_reached[b].item() / self.log_episode_goal_count[b].item() - 1) * 100)
                    self.log_goal_count[b] += self.log_episode_goal_count[b].item() - 1
                    self.log_goal_count_in_episodes.append(self.log_episode_goal_count[b].item() - 1)
                    self.log_episode_goal_count[b] = 1
                    # calculate advanatges for the full episode
                    self.update_goal_advantages(b)
                    # Since there is always a punishemnt , teacher reward is always non zero
                    self.log_teacher_reward.append(torch.mean(self.teacher_rewards[b][:self.goal_curr_ind[b]]).item())
                    self.log_proc_teacher_reward[b].append(
                        torch.mean(self.teacher_rewards[b][:self.goal_curr_ind[b]]).item())
                    self.log_intrin_teacher_reward.append(
                        torch.mean(self.intrin_teacher_rewards[b][:self.goal_curr_ind[b]]).item())
                    #calculate distinct goals
                    distinct_goals = len(
                        list(set(self.goals[b][:self.goal_curr_ind[b]].squeeze().detach().cpu().numpy())))
                    self.log_distinct_goals_epis.append(distinct_goals)
                    self.add_eps_exp_goal(b)  # Adding the episode experience for learning to the buffer
                    if self.goal_generator_info["with_step_increase"]: #Step increase is not used
                        difficulty = int(np.mean(self.log_proc_reach_steps[b][
                                                 -self.num_procs:]).item())  # difficulty = np.mean(self.log_proc_reach_steps[-self.num_procs:]) (np.mean(self.log_goal_reached[-self.num_procs:]) < self.goal_generator_info["stepi_treshhold"])

                        if (((difficulty <= self.difficulty[b]) and (
                                self.difficulty < self.goal_generator_info["difficulty_max"])) and (
                                np.mean(self.log_proc_reach[b][-self.num_procs:]).item() < self.goal_generator_info[
                            "stepi_treshold"])):
                            self.difficulty[b] += 1
                        else:
                            self.difficulty[
                                b] = difficulty  # torch.maximum(self.difficulty[b], torch.tensor(difficulty))
                    else:
                        #****** Update difficulty according to a teacher threshold ***#
                        if torch.mean(self.intrin_teacher_rewards[b][:self.goal_curr_ind[b]]).item() > \
                                self.goal_generator_info["generator_threshold"]:
                            self.difficulty_counts[b] += 1  # np.mean(self.log_teacher_return[-self.num_procs:])
                        else:
                            self.difficulty_counts[b] = 0
                        if self.difficulty_counts[b] >= self.goal_generator_info["difficulty_counts"] and \
                                self.difficulty[b] <= self.goal_generator_info["difficulty_max"]:
                            self.difficulty[b] += 1
                            self.difficulty_counts[b] = 0

                    self.reset_eps_proc(b)

            # If episode ended , initialize log episode
            # based on framework of https://github.com/lcswillems/torch-ac
            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask
            # # this is is new.
            self.log_episode_goal_reached *= self.mask
            self.log_episode_reach_steps *= self.mask

        # At the end of currrent rollout - calculate advanatages and statistics
        # Get next values for clculateing advanatges
        # This is based on framework of https://github.com/lcswillems/torch-ac
        next_obs = deepcopy(self.obs)
        # # this is is new. calculate logs for goal values
        next_obs, _, goal, _, next_goal_values = self.update_goals(next_obs, update=False)

        preporocessed_curr_obs_model = self.preprocess_obss(next_obs)
        preporocessed_curr_obs_model.image = preporocessed_curr_obs_model.image.to(self.device)
        goal = goal.to(self.device)  # this is changed from original (adding the goal)
        # This is based on framework of https://github.com/lcswillems/torch-ac but with changes to add the goal
        with torch.no_grad():
            if self.acmodel.recurrent:
                _, next_value, _ = self.acmodel(preporocessed_curr_obs_model, goal,
                                                self.memory.to(self.device) * self.mask.unsqueeze(1).to(self.device))
            else:
                _, next_value, _ = self.acmodel(preporocessed_curr_obs_model, goal=goal)
        # This is based on framework of https://github.com/lcswillems/torch-ac
        #Calculating advanatges for the student
        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i + 1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i + 1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i + 1] if i < self.num_frames_per_proc - 1 else 0
            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Define student experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.
        # This is from the framework of https://github.com/lcswillems/torch-ac

        # Define student collected experiences:
        # This is based on framework of https://github.com/lcswillems/torch-ac
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
        exps.init_obs = self.init_obss.transpose(0, 1).flatten(0, 1).to(self.device)
        exps.obs = self.preprocess_obss(exps.obs, device=self.device)
        exps.carried_obj = exps.obs.carried_obj.clone()
        exps.carried_col = exps.obs.carried_col.clone()
        exps.goal = self.student_goal.transpose(0, 1).reshape(-1).to(self.device)

        # Log some values

        keep_stat = self.log_num_return * self.num_procs
        keep = max(self.log_done_counter, keep_stat)
        # This is based on framework of https://github.com/lcswillems/torch-ac
        # But changed to add additional goal logs
        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "goal_reached_per_episode": self.log_goal_reached[-keep:],
            "teacher_mean_reward": self.log_teacher_reward[-keep:],
            "difficulty": int(torch.mean(self.difficulty).item()),
            "goal_count": np.mean(self.log_goal_count_in_episodes[-keep:]),
            "distinct_goal_epis": np.mean(self.log_distinct_goals_epis[-keep:])
        }

        # This is based on framework of https://github.com/lcswillems/torch-ac
        # But changed to add additional logs
        self.log_done_counter = 0
        self.log_return = self.log_return[-keep_stat:]
        self.log_reshaped_return = self.log_reshaped_return[-keep_stat:]
        self.log_num_frames = self.log_num_frames[-keep_stat:]
        self.log_goal_reached = self.log_goal_reached[-keep_stat:]
        self.log_teacher_return = self.log_teacher_reward[-keep_stat:]
        self.log_goal_count_in_episodes = self.log_goal_count_in_episodes[-keep_stat:]
        self.log_distinct_goals_epis = self.log_distinct_goals_epis[-keep_stat:]

        return exps, logs, self.next_exps_goals

    @abstractmethod
    def update_parameters(self):
        pass

    def update_goals(self, obs, update=True):
        # Update goals. Goals are suggested at each step (They can also be the same goals)
        # this part of the code is new

        goal_inf_list = []
        if self.goal_generator is not None:
            self.goal_generator.eval()
            new_goal = np.zeros(self.num_procs, dtype=np.float32)
            preprocessed_obs = self.preprocess_obss(obs)
            # teacher_quality_goal=torch.zeros(self.num_procs,requires_grad=False)
            with torch.no_grad(): #the teacher suggest goals
                if self.goal_generator.recurrent:
                    s_goals, s_goal_log_probs, s_goal_values, s_goal_distr, s_g_memory = self.goal_generator(
                        preprocessed_obs.image.clone().to(self.device),
                        init_obs=preprocessed_obs.init_image.clone().to(self.device),
                        diff=preprocessed_obs.diff.clone().to(self.device),
                        memory=self.goal_memory.to(self.device) * self.mask.unsqueeze(1).to(self.device),
                        carried_col=preprocessed_obs.carried_col.clone().to(self.device),
                        carried_obj=preprocessed_obs.carried_obj.clone().to(self.device), return_distribution=True)
                else:
                    s_goals, s_goal_log_probs, s_goal_values, s_goal_distr, _ = self.goal_generator(
                        preprocessed_obs.image.clone().to(self.device),
                        init_obs=preprocessed_obs.init_image.clone().to(self.device),
                        diff=preprocessed_obs.diff.clone().to(self.device),
                        carried_col=preprocessed_obs.carried_col.clone().to(self.device),
                        carried_obj=preprocessed_obs.carried_obj.clone().to(self.device), return_distribution=True)

                goal = s_goals.detach().cpu()
                goal_log_prob = s_goal_log_probs.detach().cpu()
                goal_value = s_goal_values.detach().cpu()
                #Collected the observation data
                goal_obs = preprocessed_obs.image.clone()
                goal_diff = preprocessed_obs.diff.clone()
                goal_step = preprocessed_obs.episode_step.clone()
                carried_obj = preprocessed_obs.carried_obj.clone()
                carried_col = preprocessed_obs.carried_col.clone()
                goal_init_image = preprocessed_obs.init_image.clone()

                if self.goal_generator.recurrent:
                    goal_memory = s_g_memory.detach().clone().cpu()
                obs = reset_obs(obs)  # delete goal information from next observation to prepare for next step
                # Prepare goal list to update the enviroenmnt
                for proc in range(self.num_procs):
                    goal_inf_list.append({"goal": goal[proc].item(),
                                          "goal_frame": goal_obs[proc].cpu().numpy(),
                                          "goal_new": new_goal[proc],
                                          "goal_diff": goal_diff[proc].item()
                                          })
            if update:
                self.goal = goal
                self.goal_log_prob = goal_log_prob
                self.goal_value = goal_value
                self.goal_obs = goal_obs
                self.goal_diff = goal_diff
                self.goal_step = goal_step
                self.goal_memory = goal_memory

                #add new goals to episode buffers
                for proc in range(self.num_procs):
                    self.goals[proc][self.goal_curr_ind[proc]] = self.goal[proc].item()
                    self.goal_values[proc][self.goal_curr_ind[proc]] = self.goal_value[proc].item()
                    self.goal_log_probs[proc][self.goal_curr_ind[proc]] = self.goal_log_prob[proc].item()
                    self.goal_obss[proc][self.goal_curr_ind[proc]] = self.goal_obs[proc].detach().clone()
                    # goal diffs are the diff location that the agent visited when the goal was decided
                    self.goal_diffs[proc][self.goal_curr_ind[proc]] = self.goal_diff[proc].item()
                    if self.goal_generator.recurrent:
                        self.goal_memories[proc][self.goal_curr_ind[proc]] = self.goal_memory[proc].detach().clone()
                    self.goal_steps[proc][self.goal_curr_ind[proc]] = self.goal_step[proc].item()
                    self.goals_carried_obj[proc][self.goal_curr_ind[proc]] = carried_obj[proc].item()
                    self.goals_carried_col[proc][self.goal_curr_ind[proc]] = carried_col[proc].item()
                    self.goals_init_obs[proc][self.goal_curr_ind[proc]] = goal_init_image[proc].detach().clone()

                self.goal_curr_ind += torch.ones(self.num_procs, device=self.device).to(
                    torch.long)  # Increase episode step index (at each new goal)

        return obs, goal_inf_list, goal, goal_log_prob, goal_value

    def update_hinsight_reward_teacher(self, external_reward, preprocessed_obs, curr_obs, done):
        # Update teacher rewards backwards for reached goal. If reached goal all steps that suggested it recieve a reward according to their step.
        # this part of the code is original
        # This part is new.

        for reshaped_reward in self.reshape_reward:
            if (reshaped_reward["type"] == "teacher_goal_reward"):
                reshape_teacher_func = reshaped_reward

        deleted_goal = np.full(self.num_procs, -1)  # If any goals were reached we delete them from enviroenmnt. this is to hold the deleted goals

        for proc in range(self.num_procs):
            eps_length = self.goal_curr_ind[proc]
            #If any goals s reached , add techer rewards
            if ((curr_obs.reached_goal[proc].item() > 0) or (curr_obs.last_e_reached[proc].item() > 0)):
                for ind in range(eps_length):
                    if self.intrin_teacher_rewards[proc][
                        ind] == 0.:  # If any reward was not given for the goal, then it is a potential goals to be reached
                        if self.goals[proc][ind].item() == curr_obs.goal[proc].item(): #check if it is the same goal
                            self.goals_reached[proc][ind] = 1
                            self.goal_reached_masks[proc][ind] = 0
                            deleted_goal[proc] = curr_obs.goal[proc] #the go
                            curr_teacher_intrin_reward = 0
                            number_steps_from_g = preprocessed_obs.episode_step[proc].item() + 1 - \
                                                  self.goal_steps[proc][ind].item()
                            self.reach_steps[proc][ind] = number_steps_from_g
                            proc_r = 0
                            if not done[proc]: #Adding the teacher reward
                                proc_r = reshape_teacher_func["func"](number_steps_from_g,
                                                                      reshape_teacher_func["rewards_inf"],
                                                                      self.difficulty[proc],
                                                                      curr_obs.reached_goal[proc].item() > 0,
                                                                      curr_obs.diff_type[proc].item())

                            if done[proc]:
                                proc_r = reshape_teacher_func["func"](number_steps_from_g,
                                                                      reshape_teacher_func["rewards_inf"],
                                                                      self.difficulty[proc],
                                                                      curr_obs.last_e_reached[proc].item() > 0,
                                                                      curr_obs.diff_type[proc].item())

                            curr_teacher_intrin_reward += proc_r
                            # Clamp the intrin reward between 1 and -1
                            curr_teacher_reshaped_reward = torch.clamp(torch.tensor(curr_teacher_intrin_reward), -1, 1).item()
                            self.intrin_teacher_rewards[proc][ind] = curr_teacher_reshaped_reward
                            # If the external reward is positive add 1 to the external reward and do not count the intrin reward
                            self.teacher_rewards[proc][ind] = 1. * float(external_reward[proc].detach().item() > 0) + \
                                                              self.intrin_teacher_rewards[proc][ind].item() * float(
                                external_reward[proc].detach().item() <= 0) + external_reward[proc].detach().item()

            if done[proc]:  # If the end of episode , we punish goals that were not reached, unless external goal is reached
                for ind in range(eps_length):
                    if ((external_reward[proc].detach().item() <= 0) and (not self.goals_reached[proc][ind] > 0)):
                        if not reshape_teacher_func["rewards_inf"][
                            "no_punish"]:  # If the goal was not reached at the end of episode, give negative reward
                            self.intrin_teacher_rewards[proc][ind] = -reshape_teacher_func["rewards_inf"]["beta"]
                            self.teacher_rewards[proc][ind] = self.intrin_teacher_rewards[proc][ind]
            self.env.delete_goal(deleted_goal)  # Delete the reached goals from the environemnt

    def update_goal_advantages(self, proc):
        # Update goal advanatges. this is calculated at the end of the episode
        # this part of the code is original
        # This part is new.

        eps_length = self.goal_curr_ind[proc].item()
        self.goal_masks[proc][eps_length] = 0
        self.goal_masks[proc][0] = 0
        self.goal_reached_masks[proc][0] = 0.
        self.goal_reached_masks[proc][eps_length] = 0.
        # The last value will be the bootstrp baseline. As in Amigo.
        # next_goal_value=self.goal_values[proc][eps_length-1]

        # this part of the code is new (but with the same format as torch-ac)
        for i in reversed(range(eps_length)):
            next_mask = self.goal_masks[proc][i + 1]  # if i < eps_length - 1 else next_goal_value
            next_goal_value = self.goal_values[proc][i + 1] if i < eps_length - 1 else 0
            next_goal_advantage = self.goal_advantages[proc][i + 1] if i < eps_length - 1 else 0
            delta_goal = self.teacher_rewards[proc][i] + self.discount * next_goal_value * next_mask - \
                         self.goal_values[proc][i]
            self.goal_advantages[proc][
                i] = delta_goal + self.discount * self.gae_lambda * next_goal_advantage * next_mask

        # Update goal returns
        self.goal_returns[proc] = self.goal_values[proc] + self.goal_advantages[proc]

    def reset_eps_proc(self, proc):
        # Reset episode buffer after episode has terminated
        # This part of the code is new

        self.goals[proc] = torch.full((self.env.envs[0].max_steps + 1,), -1)
        self.goal_steps[proc] *= 0
        self.goal_values[proc] *= 0
        self.goal_log_probs[proc] *= 0
        self.goal_rewards[proc] *= 0
        self.goal_diffs[proc] *= 0
        self.teacher_rewards[proc] *= 0
        self.intrin_teacher_rewards[proc] *= 0
        self.goal_advantages[proc] *= 0
        self.goal_obss[proc] *= 0
        self.goals_reached[proc] *= 0
        if self.goal_generator.recurrent:
            self.goal_memories[proc] *= 0
        self.reach_steps[proc] *= 0
        self.goal_returns[proc] *= 0
        self.goal_masks[proc] = torch.ones(self.env.envs[0].max_steps + 1)
        self.goals_carried_obj[proc] *= 0
        self.goals_carried_col[proc] *= 0
        self.goals_init_obs[proc] *= 0
        self.goal_curr_ind[proc] = 0
        self.goal_reached_masks = torch.ones(self.num_procs, self.env.envs[0].max_steps + 1)

    def add_eps_exp_goal(self, proc):
        # Add collected experiences to a buffer to be transfered to optimization
        # This part of the code is new
        next_exps_goals = None

        curr_ind = self.goal_curr_ind[proc]

        if self.exps_goals is not None:
            self.exps_goals.goal_obs = torch.cat(
                (self.exps_goals.goal_obs, self.goal_obss[proc].to("cpu").clone()[:curr_ind]), dim=0)
            self.exps_goals.goal_value = torch.cat(
                (self.exps_goals.goal_value, self.goal_values[proc].to("cpu").clone()[:curr_ind]), dim=0)
            self.exps_goals.goal_advantage = torch.cat(
                (self.exps_goals.goal_advantage, self.goal_advantages[proc].to("cpu").clone()[:curr_ind]), dim=0)
            self.exps_goals.goal = torch.cat((self.exps_goals.goal, self.goals[proc].to("cpu").clone()[:curr_ind]),
                                             dim=0)
            self.exps_goals.goal_returnn = torch.cat(
                (self.exps_goals.goal_returnn, self.goal_returns[proc].to("cpu").clone()[:curr_ind]), dim=0)
            self.exps_goals.goal_diff = torch.cat(
                (self.exps_goals.goal_diff, self.goal_diffs[proc].to("cpu").clone()[:curr_ind]), dim=0)
            self.exps_goals.goal_log_prob = torch.cat(
                (self.exps_goals.goal_log_prob, self.goal_log_probs[proc].to("cpu").clone()[:curr_ind]), dim=0)
            self.exps_goals.goal_memory = torch.cat(
                (self.exps_goals.goal_memory, self.goal_memories[proc].to("cpu").clone()[:curr_ind]), dim=0)
            self.exps_goals.mask = torch.cat(
                (self.exps_goals.mask, self.goal_masks[proc].to("cpu").clone().unsqueeze(1).to("cpu")[:curr_ind]),
                dim=0)
            self.exps_goals.carried_obj = torch.cat(
                (self.exps_goals.carried_obj, self.goals_carried_obj[proc].to("cpu").clone()[:curr_ind]), dim=0)
            self.exps_goals.carried_col = torch.cat(
                (self.exps_goals.carried_col, self.goals_carried_col[proc].to("cpu").clone()[:curr_ind]), dim=0)
            self.exps_goals.init_obs = torch.cat(
                (self.exps_goals.init_obs, self.goals_init_obs[proc].to("cpu").clone()[:curr_ind]), dim=0)

        else:
            self.exps_goals = DictList()
            self.exps_goals.goal_obs = self.goal_obss[proc].to("cpu").clone()[:curr_ind]
            self.exps_goals.goal_value = self.goal_values[proc].to("cpu").clone()[:curr_ind]
            self.exps_goals.goal_advantage = self.goal_advantages[proc].to("cpu").clone()[:curr_ind]
            self.exps_goals.goal = self.goals[proc].to("cpu").clone()[:curr_ind]
            self.exps_goals.goal_returnn = self.goal_returns[proc].to("cpu").clone()[:curr_ind]
            self.exps_goals.goal_diff = self.goal_diffs[proc].to("cpu").clone()[:curr_ind]
            self.exps_goals.goal_log_prob = self.goal_log_probs[proc].to("cpu").clone()[:curr_ind]
            self.exps_goals.goal_memory = self.goal_memories[proc].to("cpu").clone()[:curr_ind]
            self.exps_goals.mask = self.goal_masks[proc].to("cpu").clone().unsqueeze(1)[:curr_ind]
            self.exps_goals.carried_obj = self.goals_carried_obj[proc].to("cpu").clone()[:curr_ind]
            self.exps_goals.carried_col = self.goals_carried_col[proc].to("cpu").clone()[:curr_ind]
            self.exps_goals.init_obs = self.goals_init_obs[proc].to("cpu").clone()[:curr_ind]
            # self.exps_goals=curr_exp_goals

        num_batch = self.goal_num_frames

        #Prepare for transfer to optimization.
        if ((len(self.exps_goals.goal) >= self.goal_num_frames) and (self.next_exps_goals is None)):
            self.next_exps_goals = DictList()

            #add to next buffer
            self.next_exps_goals.goal_obs = self.exps_goals.goal_obs[:num_batch].to(self.device)
            self.next_exps_goals.goal_value = self.exps_goals.goal_value[:num_batch].to(self.device)
            self.next_exps_goals.goal_advantage = self.exps_goals.goal_advantage[:num_batch].to(self.device)
            self.next_exps_goals.goal = self.exps_goals.goal[:num_batch].to(self.device)
            self.next_exps_goals.goal_returnn = self.exps_goals.goal_returnn[:num_batch].to(self.device)
            self.next_exps_goals.goal_diff = self.exps_goals.goal_diff[:num_batch].to(self.device)
            self.next_exps_goals.goal_log_prob = self.exps_goals.goal_log_prob[:num_batch].to(self.device)
            self.next_exps_goals.goal_memory = self.exps_goals.goal_memory[:num_batch].to(self.device)
            self.next_exps_goals.mask = self.exps_goals.mask[:num_batch].to(self.device)
            self.next_exps_goals.carried_obj = self.exps_goals.carried_obj[:num_batch].to(self.device)
            self.next_exps_goals.carried_col = self.exps_goals.carried_col[:num_batch].to(self.device)
            self.next_exps_goals.init_obs = self.exps_goals.init_obs[:num_batch].to(self.device)

            self.exps_goals.goal_obs = self.exps_goals.goal_obs[num_batch:]
            self.exps_goals.goal_value = self.exps_goals.goal_value[num_batch:]
            self.exps_goals.goal_advantage = self.exps_goals.goal_advantage[num_batch:]
            self.exps_goals.goal = self.exps_goals.goal[num_batch:]
            self.exps_goals.goal_returnn = self.exps_goals.goal_returnn[num_batch:]
            self.exps_goals.goal_diff = self.exps_goals.goal_diff[num_batch:]
            self.exps_goals.goal_log_prob = self.exps_goals.goal_log_prob[num_batch:]
            self.exps_goals.goal_memory = self.exps_goals.goal_memory[num_batch:]
            self.exps_goals.mask = self.exps_goals.mask[num_batch:]
            self.exps_goals.carried_obj = self.exps_goals.carried_obj[num_batch:]
            self.exps_goals.carried_col = self.exps_goals.carried_col[num_batch:]
            self.exps_goals.init_obs = self.exps_goals.init_obs[num_batch:]

        return self.next_exps_goals
