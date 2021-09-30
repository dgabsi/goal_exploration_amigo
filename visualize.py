import argparse
import time
import numpy
import torch

import utils_goal_multiple


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=1000000,
                    help="number of episodes to visualize")
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

# Load environment

env = utils_goal_multiple.make_env(args.env, args.seed)
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent

model_dir = utils_goal_multiple.get_model_dir(args.model)
agent = utils_goal_multiple.Agent(env.observation_space, env.action_space, model_dir, goal_generator=True,
                                  device=device, argmax=args.argmax, use_memory=args.memory, use_text=args.text, use_amigo=args.with_amigo_nets)
print("Agent loaded\n")

#if args.goal_generator:
#    env.update_goal_generator(agent.goal_generator)

# Run the agent

if args.gif:
   from array2gif import write_gif
   frames = []

# Create a window to view the environment
env.render('human')

for episode in range(args.episodes):
    obs = env.reset()
    obs, goal_data= agent.update_goals(obs)
    env.update_goal(goal_data[0])

    while True:

        #if ((obs["goal"] is None) or (obs["goal"]<0)):
        #    goal=False
        #else:
        goal=obs["goal"]

        env.render('human', goal=int(goal))
        if args.gif:
            frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))


        action = agent.get_action(obs)
        obs, reward, done, _ = env.step(action)

        updated_obs, goal_data= agent.update_goals(obs)
        env.update_goal(goal_data[0])


        obs=updated_obs

        agent.analyze_feedback(reward, done)

        if done or env.window.closed:
            agent.reset()
            break

    if env.window.closed:
        break

if args.gif:
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
    print("Done.")
