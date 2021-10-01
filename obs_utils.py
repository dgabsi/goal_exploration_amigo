import torch
import numpy as np
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX, DIR_TO_VEC, TILE_PIXELS, Grid, WorldObj


#Various functions to analyse and render and observation
def analyse_image_atr(image):
    #Return agent and external goal location

    image_minigrid_f=image.detach().clone().numpy().astype(np.int32)
    goal_location= np.where(image_minigrid_f[:, :, 0]==OBJECT_TO_IDX['goal'])
    if len(goal_location[0])>0:
        goal_location=(goal_location[0][0],goal_location[1][0])
    agent_location=np.where(image_minigrid_f[:, :, 0]==OBJECT_TO_IDX['agent'])
    agent_location=(agent_location[0][0],agent_location[1][0])

    return image_minigrid_f, goal_location,agent_location

def render_image(image):
    #Rendering an image image_minigrid
    #The code is based on code from https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/minigrid.py
    #but changed to aff goal

    tile_size=TILE_PIXELS
    image_minigrid_f, goal_location, agent_location = analyse_image_atr(image)
    agent_pos = agent_location
    agent_dir = 0
    highlight_mask = None

    width, height=image_minigrid_f.shape[:2]

    if highlight_mask is None:
        highlight_mask = np.zeros(shape=(width, height), dtype=np.bool)

    # Compute the total grid size
    width_px = width * tile_size
    height_px = height * tile_size

    img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

    # Render the grid
    for i in range(0, height):
        for j in range(0, width):
            obj=int(image_minigrid_f[i, j, 0])
            obj=(obj if obj!=10 else 0)
            cell = WorldObj.decode(obj, int(image_minigrid_f[i, j, 1]), int(image_minigrid_f[i, j, 2]))

            agent_here = ((agent_pos[0]==i) and (agent_pos[1]==j))
            tile_img = Grid.render_tile(
                cell,
                agent_dir=agent_dir if agent_here else None,
                highlight=highlight_mask[i, j],
                tile_size=tile_size
            )

            ymin = i * tile_size
            ymax = (i + 1) * tile_size
            xmin = j * tile_size
            xmax = (j + 1) * tile_size
            img[ymin:ymax, xmin:xmax, :] = tile_img

    return img

def reset_obs(obs):
    #Reset the observation and delete goals information

    for proc in range(len(obs)):
        obs[proc]["goal"] = -1
        obs[proc]["goal_image"] = np.full_like(obs[proc]["init_image"],-1)
        obs[proc]["goal_diff"] = -1
        obs[proc]["goal_step"] = -1
        obs[proc]["reached_goal"] = 0
        obs[proc]["last_e_reached"] = 0
        obs[proc]["last_e_step"] = -1
        obs[proc]["last_e_r_weight"] = 0

    return obs