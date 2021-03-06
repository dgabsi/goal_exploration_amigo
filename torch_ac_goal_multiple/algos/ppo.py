import numpy
import torch
import torch.nn.functional as F

from torch_ac_goal_multiple.algos.base import BaseAlgo

#the code is an implementation of AMIGo multiple
#For AMIGo paper:
#Learning with AMIGo: Adversarially Motivated Intrinsic GOals
#(Campero et al., 2015)(https://arxiv.org/abs/2006.12122))

#The implementation is based on the RL torch-ac framework available at https://github.com/lcswillems/torch-ac
#but was changed considerably to add AMIGo capabilities
# as refernce for AMIGo I used  https://github.com/facebookresearch/adversarially-motivated-intrinsic-goals
class PPOAlgo(BaseAlgo):
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None, log_num_return=50, goal_generator=None, goal_generator_args=None, image_model=None, goal_recurrence=1, archive_args=None, total_frames=None):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, log_num_return, goal_generator, goal_generator_args, image_model, goal_recurrence, archive_args)

        self.clip_eps = clip_eps
        self.epochs = epochs

        self.total_frames = total_frames
        self.processes = len(envs)

        #Rhis is based on https://github.com/lcswillems/torch-ac
        if self.acmodel is not None:
            self.batch_size = batch_size
            assert self.batch_size % self.recurrence == 0
            self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
            self.batch_num = 0
            self.batch_size = batch_size
            #self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_lambda)

        # This part is new-creating optimizer for the goal generator (teacher). New
        if self.goal_generator is not None:
            assert self.goal_batch_size % self.goal_recurrence == 0
            self.goal_batch_num = 0
            self.goal_optimizer = torch.optim.Adam(self.goal_generator.parameters(), self.goal_generator_info["lr"], eps=adam_eps)
            #self.generator_scheduler = torch.optim.lr_scheduler.LambdaLR(self.goal_optimizer, self.lr_lambda)

    def update_parameters(self, exps, exps_goals, iter):
        # Update parameters- train using PPO algorithm

        self.goal_generator = self.goal_generator.to(self.device)
        self.goal_generator.train()
        self.acmodel.train()

        for _ in range(self.epochs):
            # Initialize log values
            #this is from https://github.com/lcswillems/torch-ac
            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            # Adding the goal loss attributes- this is new
            log_goal_entropies = []
            log_goal_values = []
            log_goal_policy_losses = []
            log_goal_value_losses = []
            log_goal_grad_norms = []

            # this is from https://github.com/lcswillems/torch-ac
            for inds in self._get_batches_starting_indexes():
                # Initialize batch values
                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                #This is new
                #initalise batch of goals
                goal_batch_entropy = 0
                goal_batch_value = 0
                goal_batch_policy_loss = 0
                goal_batch_value_loss = 0
                goal_batch_loss = 0

                # Initialize memory

                if self.acmodel.recurrent:
                    memory = exps.memory[inds]
                if (exps_goals is not None):
                    if self.goal_generator.recurrent:
                        goal_memory = exps_goals.goal_memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience
                    # this is from https://github.com/lcswillems/torch-ac
                    sb = exps[inds + i]
                    # Compute loss
                    if self.acmodel.recurrent:
                        dist, value, memory= self.acmodel(sb.obs, sb.goal, memory=memory * sb.mask)
                    else:
                        dist, value, _ = self.acmodel(sb.obs, sb.goal)
                    entropy = dist.entropy().mean()
                    #ppo policy ratio
                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    #critic loss
                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()
                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    # Update batch values
                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch

                    if self.acmodel.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                    #This part is new -adding PPO optimization also to the goal generator(teacher).
                    # (whithin the general from as as https://github.com/lcswillems/torch-ac. but updated to goals
                    if (exps_goals is not None):
                        sb_goals = exps_goals[inds + i]

                        if self.goal_generator.recurrent:
                            _, _, goal_value, goal_dist, goal_memory = self.goal_generator(sb_goals.goal_obs, return_distribution=True,
                                                                              init_obs=sb_goals.init_obs, diff=sb_goals.goal_diff, memory=goal_memory * sb_goals.mask, carried_col=sb_goals.carried_col.clone(), carried_obj=sb_goals.carried_obj.clone())
                        else:
                            _, _, goal_value, goal_dist = self.goal_generator(sb_goals.goal_obs, return_distribution=True,
                                                                        init_obs=sb_goals.init_obs, diff=sb_goals.diff, carried_col=sb_goals.carried_col.clone(), carried_obj=sb_goals.carried_obj.clone())
                        goal_entropy = goal_dist.entropy().mean()
                        goal_ratio = torch.exp(goal_dist.log_prob(sb_goals.goal) - sb_goals.goal_log_prob)
                        goal_surr1 = goal_ratio * sb_goals.goal_advantage
                        goal_surr2 = torch.clamp(goal_ratio, 1.0 - self.clip_eps,1.0 + self.clip_eps) * sb_goals.goal_advantage
                        goal_policy_loss = -torch.min(goal_surr1, goal_surr2).mean()

                        goal_value_clipped = sb_goals.goal_value + torch.clamp(goal_value - sb_goals.goal_value, -self.clip_eps, self.clip_eps)
                        goal_surr1 = (goal_value - sb_goals.goal_returnn).pow(2)
                        goal_surr2 = (goal_value_clipped - sb_goals.goal_returnn).pow(2)
                        goal_value_loss = torch.max(goal_surr1, goal_surr2).mean()

                        goal_loss = goal_policy_loss - self.goal_generator_info["entropy_coef"] * goal_entropy + \
                                        self.goal_generator_info["value_coef"] * goal_value_loss

                        goal_batch_entropy += goal_entropy.item()
                        goal_batch_value += goal_value.mean().item()
                        goal_batch_policy_loss += goal_policy_loss.item()
                        goal_batch_value_loss += goal_value_loss.item()
                        goal_batch_loss += goal_loss

                        if self.goal_generator.recurrent and i < self.goal_recurrence - 1:
                            exps_goals.goal_memory[inds + i + 1] = goal_memory.detach()

                # this is from https://github.com/lcswillems/torch-ac
                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence
                # Update actor-critic
                # this is from https://github.com/lcswillems/torch-ac
                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values# this is from https://github.com/lcswillems/torch-ac
                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)

                # This part is new -adding PPO optimization also to the goal generator(teacher).
                # this is in the same form as https://github.com/lcswillems/torch-ac. but updated to goals
                if (exps_goals is not None):
                    goal_batch_entropy /= self.goal_recurrence
                    goal_batch_value /= self.goal_recurrence
                    goal_batch_policy_loss /= self.goal_recurrence
                    goal_batch_value_loss /= self.goal_recurrence
                    goal_batch_loss /= self.goal_recurrence

                    # Train goal generator
                    self.goal_optimizer.zero_grad()
                    goal_batch_loss.backward()
                    goal_grad_norm = sum(
                        p.grad.data.norm(2).item() ** 2 for p in self.goal_generator.parameters()) ** 0.5
                    torch.nn.utils.clip_grad_norm_(self.goal_generator.parameters(), self.max_grad_norm)
                    self.goal_optimizer.step()

                    log_goal_entropies.append(goal_batch_entropy)
                    log_goal_values.append(goal_batch_value)
                    log_goal_policy_losses.append(goal_batch_policy_loss)
                    log_goal_value_losses.append(goal_batch_value_loss)
                    log_goal_grad_norms.append(goal_grad_norm)

            #self.scheduler.step()
            #if (exps_goals is not None):
            #    self.generator_scheduler.step()

        # Log some values
        ## this is from https://github.com/lcswillems/torch-ac but I added goal logs
        logs = {
            "entropy": 0. if len(log_entropies)==0  else numpy.mean(log_entropies),
            "value": 0. if len(log_entropies)==0 else numpy.mean(log_values),
            "policy_loss": 0. if len(log_entropies)==0 else numpy.mean(log_policy_losses),
            "value_loss": 0. if len(log_entropies)==0 else numpy.mean(log_value_losses),
            "grad_norm": 0. if len(log_entropies)==0 else numpy.mean(log_grad_norms),
            "goal_entropy": 0. if len(log_goal_entropies)==0 else numpy.mean(log_goal_entropies),
            "goal_value": 0. if len(log_goal_entropies)==0 else numpy.mean(log_goal_values),
            "goal_policy_loss": 0. if len(log_goal_entropies)==0 else numpy.mean(log_goal_policy_losses),
            "goal_value_loss": 0. if len(log_goal_entropies)==0 else numpy.mean(log_goal_value_losses),
            "goal_grad_norm": 0. if len(log_goal_entropies)==0 else numpy.mean(log_goal_grad_norms)
        }

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """
        ## this is from https://github.com/lcswillems/torch-ac
        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1
        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes

    def _get_goals_batches_starting_indexes(self):

        # This part is new -adding PPO optimization also to the goal generator(teacher).
        # (whithin the general from as as https://github.com/lcswillems/torch-ac. but updated to goals
        indexes = numpy.arange(0, self.num_frames, self.goal_recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.goal_batch_num % 2 == 1:
            indexes = indexes[(indexes + self.goal_recurrence) % self.num_frames_per_proc != 0]
            indexes += self.recurrence // 2
        self.goal_batch_num += 1
        num_indexes = self.goal_batch_size // self.goal_recurrence
        batches_starting_indexes = [indexes[i:i + num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes

    def lr_lambda(self, epoch):
        return 1 - (epoch* self.num_frames_per_proc*self.num_procs/ self.total_frames)