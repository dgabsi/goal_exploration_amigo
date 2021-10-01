import gym
import gym_minigrid
from gym_minigrid.wrappers import FullyObsWrapper
from gym.wrappers import Monitor
import torch
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX, TILE_PIXELS
import numpy as np
from goal_generator_model import GoalGeneratorModel

def _format_observation(obs):
    #obs = np.long(obs)
    return obs #np.expand_dims(obs,0)


#this is a new enviornment wrapper. this part of the code is new,
#The general stucture is based on wrapper of Amigo, but almost all is changed or added.https://github.com/facebookresearch/adversarially-motivated-intrinsic-goals
class Observation_WrapperMiniGrid(gym.core.Wrapper):
    """
        Wrapper that add to the environemnts the capability to holdgoals, check if a goals has been reached.
        It also calculates diff_location, which is the location that have been changed at each step
        """

    def __init__(self, env, env_seed=None, fix_seed=True, modify=True):
        super().__init__(env)

        # Attibutes for saving information about goals and diff location
        self.episode_return = None
        self.episode_step = 0
        self.episode_win = None
        self.fix_seed = fix_seed
        self.env_seed = env_seed
        self.last_frame=None
        self.last_diff=None

        self.goal=[]

        self.max_steps=self.max_steps
        self.goal_frame=[]
        self.reached_goal=np.array(0, dtype=np.uint8)
        self.reached_weight=0
        self.last_e_reached=np.array(0, dtype=np.uint8)
        self.last_e_step=None
        self.last_e_r_weight=0
        self.curr_goal_reached = None
        self.curr_goal_step_given = None

        self.step_goal_given=[]
        self.initial_frame=None
        self.modify=modify
        self.prev_diff=-1
        self.goal_diff=[]



    def reset(self, **kwargs):

        # If the former episode was terminated in a success , then the reset step will give the goal information. since it automaticly goes to reset.
        goal=-1
        goal_step=-1

        if self.reached_goal>0:
            self.last_e_reached = self.reached_goal
            self.last_e_step=self.episode_step+1-self.curr_goal_step_given
            self.last_e_r_weight=self.reached_weight
            goal=self.curr_goal_reached
            goal_step=self.curr_goal_step_given
        else:
            self.last_e_reached=0
            self.last_e_step =-1
            self.last_e_r_weight=0

        if self.fix_seed:
            super().seed(seed=self.env_seed)

        obs = super().reset(**kwargs) #call the reset step of previous wrapper

        initial_reward = 0
        self.episode_return = 0
        self.episode_step = 0
        self.episode_win = 0
        initial_done = 0
        initial_frame = _format_observation(obs['image']).copy()
        self.initial_frame=initial_frame.copy()

        text_obs = obs['mission']
        self.last_frame=obs['image']

        W, H, C = initial_frame.shape

        frame=self.initial_frame.copy()
        self.last_frame=self.initial_frame.copy()
        diff_weight = 0
        diff_type = 0
        diff=-1
        self.prev_diff=None

        # In the reset step there is no goal
        self.reached_goal=0
        self.goal=[]
        self.goal_frame=[]
        self.goal_diff = []
        self.step_goal_given=[]
        self.reached_weight=0
        self.curr_goal_reached=None
        self.curr_goal_step_given=None


        carried_col, carried_obj = np.long(5), np.long(1)
        if self.carrying:
            carried_col, carried_obj = np.long(COLOR_TO_IDX[self.carrying.color]), np.long(
                OBJECT_TO_IDX[self.carrying.type])


        return dict(
            image=frame.copy(),
            reward=initial_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            episode_win=self.episode_win,
            carried_col=carried_col,
            carried_obj=carried_obj,
            mission= text_obs,
            diff= int(diff),
            diff_type=diff_type,
            goal=goal,
            reached_goal=0,
            reached_weight=0,
            seed=self.env_seed,
            goal_image=np.full_like(initial_frame,-1),
            goal_step=goal_step,
            goal_diff=-1,
            last_e_reached=self.last_e_reached,
            last_e_step=self.last_e_step,
            last_e_r_weight=self.last_e_r_weight,
            init_image= initial_frame.copy())


    def step(self, action):

        frame, reward, done, info = super().step(action)

        self.episode_step += 1
        episode_step = self.episode_step

        self.episode_return = self.episode_return +reward #np.array(float(reward), dtype=np.float32)
        episode_return = self.episode_return

        text_obs = frame['mission']
        frame = _format_observation(frame['image'])
        W, H, C=frame.shape

        self.last_e_reached = 0
        self.last_e_step = -1
        self.last_e_r_weight = 0
        self.curr_goal_reached = None
        self.curr_goal_step_given = None

        # Calculate diff location and decide which is the newest diff location based on the former diff location
        diff_type = 0
        compare_frame=self.last_frame.copy()
        if compare_frame is not None:
            diff_all = diff_frame(frame, compare_frame, self.initial_frame)
            if diff_all is None:
                diff = (self.prev_diff.copy() if self.prev_diff is not None else None)
                diff_type=0 #(0 if diff is None else len(diff))
            else:
                diff_all=diff_all.squeeze(-1)
                diff_type = len(diff_all)
                if ((self.prev_diff is not None) and len(diff_all)>1):
                    diff =np.array([i for i in diff_all if i not in self.prev_diff[:]])
                else:
                    diff=diff_all
                self.prev_diff = diff.copy()
        else:
                diff = None
                self.prev_diff = None


        self.last_frame = frame.copy()

        if done and reward > 0:
            self.episode_win = 1
        else:
            self.episode_win = 0
        episode_win = self.episode_win

        self.reached_goal = 0
        reached_goal = 0
        #Check if the diff location contains the goal
        for ind, curr_goal in enumerate(self.goal):
            goal=curr_goal
            goal_frame=self.goal_frame[ind].copy()
            goal_diff = self.goal_diff[ind]
            goal_step_given = self.step_goal_given[ind]
            reached_weight=self.reached_weight
            if ((diff_type>0) and (int(goal) in diff[:])):
                reached_goal = 1
                self.reached_goal = 1
                diff=[goal]
                self.reached_weight=diff_type
                reached_weight=diff_type
                self.curr_goal_reached=goal
                self.curr_goal_step_given=self.step_goal_given[ind]
                break

        if reached_goal==0:
            goal = None
            goal_frame = None
            goal_diff = None
            goal_step_given = None
            reached_weight = 0

        if diff is not None:
            if len(diff)>=1:
                diff = int(np.random.choice(diff, 1))
        else:
            diff=-1

        reward = reward
        done = done

        carried_col, carried_obj = np.long(5), np.long(1)
        if self.carrying:
            carried_col, carried_obj = np.long(COLOR_TO_IDX[self.carrying.color]), np.long(OBJECT_TO_IDX[self.carrying.type])


        return dict(
            image=frame,
            reward=reward,
            done=done,
            episode_return=episode_return,
            episode_step=episode_step,
            episode_win=episode_win,
            carried_col=carried_col,
            carried_obj=carried_obj,
            mission= text_obs,
            diff=(int(diff)),
            diff_type=diff_type,
            goal=(goal if goal is not None else -1),
            reached_goal=reached_goal,
            reached_weight=(reached_weight if reached_goal>0 else 0),
            seed=self.env_seed,
            goal_image=(goal_frame if goal_frame is not None else np.full_like(frame,-1)),
            goal_step=(goal_step_given if goal_step_given is not None else -1),
            goal_diff=(goal_diff if goal_diff is not None else -1),
            last_e_reached=0,
            last_e_step=-1,
            last_e_r_weight=0,
            init_image= self.initial_frame.copy()
        ), reward, done, info


    def update_goal(self, goal_data):
        # Update goals data in the enviornment
        #Always updates the goals
        goal, goal_frame, _, goal_diff= goal_data["goal"],goal_data["goal_frame"],goal_data["goal_new"],goal_data["goal_diff"]

        self.step_goal_given.append(self.episode_step)
        self.reached_goal=0
        self.reached_weight = 0
        self.goal.append(goal)
        self.goal_frame.append(goal_frame.copy())
        self.goal_diff.append(goal_diff)


        return goal

    def delete_goal(self,deleted_goal):
        #Delete a goal from the environemnt
        if deleted_goal!=-1:
            for ind, curr_goal in enumerate(self.goal):
                if curr_goal==deleted_goal:
                    self.goal.pop(ind)
                    self.goal_frame.pop(ind)
                    self.goal_diff.pop(ind)
                    self.step_goal_given.pop(ind)

        return deleted_goal



    def render(self, mode='human', close=False, highlight=True, tile_size=TILE_PIXELS, goal=False):
        #Rendering an image image_minigrid
    #The code is based on code from https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/minigrid.py
    #but changed to add goal

        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            import gym_minigrid.window
            self.window = gym_minigrid.window.Window('gym_minigrid')
            self.window.show(block=False)

        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        # Compute the world coordinates of the bottom-left corner
        # of the agent's view area
        f_vec = self.dir_vec
        r_vec = self.right_vec
        top_left = self.agent_pos + f_vec * (self.agent_view_size-1) - r_vec * (self.agent_view_size // 2)

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=np.bool)

        # For each cell in the visibility mask
        for vis_j in range(0, self.agent_view_size):
            for vis_i in range(0, self.agent_view_size):
                # If this cell is not visible, don't highlight it
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.width:
                    continue
                if abs_j < 0 or abs_j >= self.height:
                    continue

                # Mark this cell to be highlighted
                highlight_mask[abs_i, abs_j] = True

        if goal:
            highlight_mask = np.zeros(shape=(self.width*self.height), dtype=np.bool)
            highlight_mask[goal]=True
            highlight_mask=np.reshape(highlight_mask, (self.width,self.height))
        else:
            a=1

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=highlight_mask if highlight else None
        )

        if mode == 'human':
            self.window.set_caption(self.mission)
            self.window.show_img(img)

        return img


def make_env(env_key, seed=None, fix_seed=True, modify=True, video_dir=None):
    env = gym.make(env_key)
    env = FullyObsWrapper(env)
    #if video_dir is not None:
       #env = Monitor(env, video_dir+'/video', force=True)
    env = Observation_WrapperMiniGrid(env, seed, fix_seed, modify)
    if fix_seed:
        env.seed(seed)
    return env


def diff_frame(new_frame, frame, initial_frame, is_area_interes=False):
    # Calculate the diff location
    # Which locations have changed in one of :object, color or state

    new_frame=new_frame.copy()
    frame=frame.copy()
    inital_frame = initial_frame.copy()
    new_obs = np.reshape(new_frame, (-1,new_frame.shape[-1]))
    prev_obs = np.reshape(frame, (-1,frame.shape[-1]))
    inital_obs = np.reshape(inital_frame, (-1,inital_frame.shape[-1]))
    diff = (new_obs == prev_obs)
    ans=(np.sum(diff, -1) != 3).astype(np.float32)
    if ans.any():
        diff_loc= np.argwhere(ans>0)#np.argmax(ans)

        sum_initial=np.sum(inital_obs,-1)
        initial_z_score=(sum_initial- np.mean(sum_initial))/np.std( sum_initial)

        a=np.abs(initial_z_score[diff_loc])
        #if np.abs(initial_z_score[diff_loc])<1:
        #    diff_loc = None
    else:
        diff_loc=None

    return diff_loc


