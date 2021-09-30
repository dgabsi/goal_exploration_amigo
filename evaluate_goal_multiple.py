import argparse
import time
import torch
from torch_ac_goal_multiple.utils.penv import ParallelEnv
import argparse
import time
import datetime
import torch_ac_goal_multiple
import tensorboardX
import sys
import pandas as pd
import os

import utils_goal_multiple


#Evaluate an agent performence. after it has been trained
# The basis of script is https://github.com/lcswillems/rl-starter-files/blob/master/scripts/
#but is has been change to add AMIGO capabilities
# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--episodes", type=int, default=100,
                    help="number of episodes of evaluation (default: 100)")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                    help="how many worst episodes to show")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")
parser.add_argument("--with_amigo-nets", action="store_true", default=False,
                    help="use amigo networks")
args = parser.parse_args()

# Set seed for all randomness sources

utils_goal_multiple.seed(args.seed)

# Set device

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
eval_default_model_name = f"{args.env}_{args.model}_seed{args.seed}_{date}_eval"

model_name = eval_default_model_name
eval_model_dir = utils_goal_multiple.get_model_dir(model_name)

# Load loggers and Tensorboard writer

txt_logger = utils_goal_multiple.get_txt_logger(eval_model_dir)
csv_file, csv_logger = utils_goal_multiple.get_csv_logger(eval_model_dir)


# Load environments

envs = []
for i in range(args.procs):
    env = utils_goal_multiple.make_env(args.env, args.seed + 10000 * i)
    envs.append(env)
env = ParallelEnv(envs)
print("Environments loaded\n")

# Load agent

model_dir = utils_goal_multiple.get_model_dir(args.model)
agent = utils_goal_multiple.Agent(env.observation_space, env.action_space, model_dir, goal_generator=True,
                                  device=device, argmax=args.argmax, num_envs=args.procs, use_memory=args.memory, use_text=args.text, use_amigo=args.with_amigo_nets)
print("Agent loaded\n")

# Initialize logs

logs = {"num_frames_per_episode": [], "return_per_episode": []}

# Run agent

start_time = time.time()

obss = env.reset()
obs, goal_data = agent.update_goals(obss)
env.update_goal(goal_data)

log_done_counter = 0
log_episode_return = torch.zeros(args.procs, device=device)
log_episode_num_frames = torch.zeros(args.procs, device=device)

while log_done_counter < args.episodes:
    #This is new. suggest goals and update the envrioenmnt
    updated_obs, goal_data = agent.update_goals(obss, argmax=args.argmax)
    env.update_goal(goal_data)
    obss = updated_obs


    actions = agent.get_actions(obss)
    obss, rewards, dones, _ = env.step(actions)


    agent.analyze_feedbacks(rewards, dones)


    log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
    log_episode_num_frames += torch.ones(args.procs, device=device)

    for i, done in enumerate(dones):
        if done:
            log_done_counter += 1
            logs["return_per_episode"].append(log_episode_return[i].item())
            logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

    mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
    log_episode_return *= mask
    log_episode_num_frames *= mask

end_time = time.time()

# Print logs

logs_data=pd.DataFrame.from_dict(logs)
logs_data.to_csv(os.path.join(eval_model_dir, "logs_data.csv"))

num_frames = sum(logs["num_frames_per_episode"])
fps = num_frames/(end_time - start_time)
duration = int(end_time - start_time)
return_per_episode = utils_goal_multiple.synthesize(logs["return_per_episode"])
num_frames_per_episode = utils_goal_multiple.synthesize(logs["num_frames_per_episode"])

header = ["num_frames_per_episode","FPS", "duration"]
header += ["rreturn_" + key for key in return_per_episode.keys()]
header += ["numframes_" + key for key in num_frames_per_episode.keys()]
data = [num_frames_per_episode, fps, duration,
              *return_per_episode.values(),
              *num_frames_per_episode.values()]


print("F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
      .format(num_frames, fps, duration,
              *return_per_episode.values(),
              *num_frames_per_episode.values()))

csv_logger.writerow(header)
csv_logger.writerow(data)
csv_file.flush()

# Print worst episodes

n = args.worst_episodes_to_show
if n > 0:
    print("\n{} worst episodes:".format(n))

    indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
    for i in indexes[:n]:
        print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))


