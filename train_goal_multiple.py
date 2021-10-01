import argparse
import time
import datetime
import torch
import torch_ac_goal_multiple
import tensorboardX
import sys
import os

import utils_goal_multiple
from model import ACModel
from goal_generator_model import GoalGeneratorModel
from amigo_generator_model import  Amigo_GoalGenerator
from amigo_ac_model import Amigo_ACModel
from reshape_reward_goal_multiple import student_goal_reward, teacher_goal_reward


# The basis of script is https://github.com/lcswillems/rl-starter-files/blob/master/scripts/
#I have added changes to include Amigo capabilities. for reference for AMIGo ,I used https://github.com/facebookresearch/adversarially-motivated-intrinsic-goals
# Parse arguments

parser = argparse.ArgumentParser()

## General parameters
#parser.add_argument("--algo", required=True,
#                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=40,
                    help="number of processes (default: 40)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")
parser.add_argument("--log-num-return", type=int, default=50,
                    help="number of episodes returns for logs")
parser.add_argument("--fix_seed", action="store_true", default=False,
                    help="fix seeds in environments")
parser.add_argument('--record_video', action='store_true',
                    help='Record video of agent')

parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--frames-per-proc", type=int, default=128,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--with_amigo_nets", action="store_true", default=False,
                    help="use amigo nets")
parser.add_argument("--hidden_size_amigo", type=int, default=256,
                    help="hidden units for amigo netowrk")


#Student agent specific args
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--entropy-coef", type=float, default=0.0005, #0.001 #0.0005
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")
parser.add_argument("--coef_student_intrin_reward", type=float, default=0.001,
                    help="student intrin reward coefficient")



#Goal Generator specific args
#based on https://github.com/facebookresearch/adversarially-motivated-intrinsic-goals
parser.add_argument("--teacher-reward-positive", type=float, default=0.007,
                    help="alpha, teacher positive reward ")
parser.add_argument("--teacher-reward-negative", type=float, default=0.001,
                    help="beta, teacher negative reward ")
parser.add_argument("--modify", action="store_true", default=False,
                    help="reward is the content of the cell changes")
parser.add_argument("--lr-teacher", type=float, default=0.002, #0.005
                    help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda-teacher", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef-teacher", type=float, default=0.005, #0.05
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef-teacher", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--goal-recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--goal_batch-size", type=int, default=256,
                    help="goal batch size")


#args for fixed increase in managing difficulty
#based on https://github.com/facebookresearch/adversarially-motivated-intrinsic-goals
parser.add_argument("--difficulty_start", type=float, default=5.,
                    help="initial value of difficulty")
parser.add_argument("--generator_threshold", type=float, default=-0.001,
                    help="Threshold mean reward for which scheduler increases difficulty")
parser.add_argument('--difficulty_counts', default=10, type=int,
                    help='Number of time before generator increases difficulty')
parser.add_argument('--difficulty_maximum', default=100, type=float,
                    help='Maximum difficulty')
parser.add_argument('--difficulty_freq', default=10, type=int,
                    help='frequency of checking difficulty')

'''#For future use
#args for steps increase in managing difficulty
parser.add_argument("--with-steps-increase", action="store_true", default=False,
                    help="calculate steps stats and increase difficulty")
parser.add_argument("--step_increase_reach_treshold", type=float, default=2,
                    help="percentage of goal reach to increase difficulty when steps increase is used")
parser.add_argument("--no_punishemnt", action="store_true", default=False,
                    help="No punishment for teacher when student didnt reach goal")

#Additional args
parser.add_argument("--with-image-memory", action="store_true", default=False,
                    help="add image memory")
parser.add_argument("--goal_frequency", type=int, default=0,
                    help="goal-frequency")
parser.add_argument("--add_archive_rewards", action="store_true", default=False,
                    help="add rewards for goals based on archive of winning episodes")
parser.add_argument("--arch_modified_reward", type=float, default=0.7,
                    help="reward for reaching a modified location from the archive")
parser.add_argument("--arch_visit_reward", type=float, default=0.,
                    help="reward for reaching a visited location from the archive")
parser.add_argument("--student_modify_reward", type=float, default=0.0,
                    help="student reward for modification of cell")
parser.add_argument("--goal-icm-reward", action='store_true',
                    help="add ICM reward")
'''

args = parser.parse_args()

args.mem = args.recurrence > 1
args.goal_mem = args.goal_recurrence > 1

# Set run dir

date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{args.env}_goal_multiple_{date}"

model_name = args.model or default_model_name
model_dir = utils_goal_multiple.get_model_dir(model_name)

# Load loggers and Tensorboard writer

txt_logger = utils_goal_multiple.get_txt_logger(model_dir)
csv_file, csv_logger = utils_goal_multiple.get_csv_logger(model_dir)
tb_writer = tensorboardX.SummaryWriter(model_dir)

# Log command and all script arguments

txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))

# Set seed for all randomness sources

utils_goal_multiple.seed(args.seed)

# Set device

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
txt_logger.info(f"Device: {device}\n")

# Load environments
video_dir=None
envs = []
video_dir=None
fix_seed=False
if args.fix_seed:
    fix_seed=True
if args.record_video:
    video_dir = model_dir
for i in range(args.procs):
    envs.append(utils_goal_multiple.make_env(args.env, args.seed + 10000 * i, fix_seed=fix_seed, modify=False, video_dir=model_dir))
txt_logger.info("Environments loaded\n")

# Load training status

try:
    status = utils_goal_multiple.get_status(model_dir)
except OSError:
    status = {"num_frames": 0, "update": 0}
txt_logger.info("Training status loaded\n")


# Load observations preprocessor

obs_space, preprocess_obss = utils_goal_multiple.get_obss_preprocessor(envs[0].observation_space, with_goal=True)
if "vocab" in status:
    preprocess_obss.vocab.load_vocab(status["vocab"])
txt_logger.info("Observations preprocessor loaded")

# Load model
unet_model=None


#Create student network
if args.with_amigo_nets:
    #Use amigo type student network
    acmodel=Amigo_ACModel(obs_space, envs[0].action_space, device, hidden_size=args.hidden_size_amigo,use_memory=args.mem, use_text=args.text)
else:
    acmodel = ACModel(obs_space, envs[0].action_space, True, args.mem, args.text)
if "model_state" in status:
    acmodel.load_state_dict(status["model_state"])
acmodel.to(device)
txt_logger.info("Model loaded\n")
txt_logger.info("{}\n".format(acmodel))

#Create teacher network
if args.with_amigo_nets:
    goal_generator_model = Amigo_GoalGenerator(obs_space, device, args.goal_mem)
else:
    goal_generator_model = GoalGeneratorModel(obs_space, device=device, use_memory=args.goal_mem)
if "goal_generator_state" in status:
    goal_generator_model.load_state_dict(status["goal_generator_state"])
goal_generator_model=goal_generator_model.to(device)
txt_logger.info("Goal generator loaded\n")
txt_logger.info("{}\n".format(goal_generator_model))
goal_generator_args={"lr": args.lr_teacher, "value_coef":args.value_loss_coef_teacher,"entropy_coef":args.entropy_coef_teacher,"with_step_increase":False,
                     "goal_frequency":0, "generator_threshold":args.generator_threshold, "difficulty": args.difficulty_start,
                     "difficulty_counts":args.difficulty_counts, "difficulty_max":args.difficulty_maximum, "goal_batch_size":args.goal_batch_size,
                     "stepi_treshhold": 2, "diff_freq":10}

#Reshaped reward function
reshape_reward=[]
#Student reshaped reward
reshape_reward.append({"func": student_goal_reward, "type": "student_goal_reward", "reward_coef":1.,
                       "rewards_inf":{"modify":args.modify, "modified_reward": False, "reward_coef":args.coef_student_intrin_reward}})
#Teacher reshaped reward
reshape_reward.append({"func": teacher_goal_reward, "type": "teacher_goal_reward", "reward_coef": 1.,
                       "rewards_inf": {"alpha": args.teacher_reward_positive, "beta":args.teacher_reward_negative,
                        "modify":False,  "no_punish":False}})

#Other archive
#archive_args={"use_archive":args.add_archive_rewards, "modified_reward":args.arch_modified_reward, "visit_reward":args.arch_visit_reward}

# Load algo


algo = torch_ac_goal_multiple.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                          args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                          args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss, reshape_reward, args.log_num_return, goal_generator_model, goal_generator_args, None, args.goal_recurrence, None, total_frames=args.frames)


#Load optimizer and schedulers
if "optimizer_state" in status:
    algo.optimizer.load_state_dict(status["optimizer_state"])
if "goal_optimizer_state" in status:
    algo.goal_optimizer.load_state_dict(status["goal_optimizer_state"])
if "scheduler_state" in status:
    algo.scheduler.load_state_dict(status["scheduler_state"])
if "goal_scheduler_state" in status:
    algo.generator_scheduler.load_state_dict(status["goal_scheduler_state"])

txt_logger.info("optimizer loaded\n")

# Train model
num_frames = status["num_frames"]
update = status["update"]
start_time = time.time()

while num_frames < args.frames:
    # Update model parameters

    update_start_time = time.time()
    exps, logs1, exps_goal = algo.collect_experiences()

    logs2 = algo.update_parameters(exps, exps_goal, num_frames)
    logs = {**logs1, **logs2}
    update_end_time = time.time()

    num_frames += logs["num_frames"]
    update += 1

    # Print logs

    if update % args.log_interval == 0:
        fps = logs["num_frames"]/(update_end_time - update_start_time)
        duration = int(time.time() - start_time)
        return_per_episode = utils_goal_multiple.synthesize(logs["return_per_episode"])
        rreturn_per_episode = utils_goal_multiple.synthesize(logs["reshaped_return_per_episode"])
        num_frames_per_episode = utils_goal_multiple.synthesize(logs["num_frames_per_episode"])


        header = ["update", "frames", "FPS", "duration"]
        data = [update, num_frames, fps, duration]
        header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
        data += rreturn_per_episode.values()
        header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
        data += num_frames_per_episode.values()
        header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
        data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]
        header += ["return_" + key for key in return_per_episode.keys()]
        data += return_per_episode.values()

        teacher_mean_reward = utils_goal_multiple.synthesize(logs["teacher_mean_reward"])
        reached_per_episode = utils_goal_multiple.synthesize(logs["goal_reached_per_episode"])
        distinct_goal_epis = utils_goal_multiple.synthesize(logs["distinct_goal_epis"])
        header += ["teach_return_mean", "difficulty", "goal_count", "dis_goals"]
        data += [teacher_mean_reward["mean"], logs["difficulty"], logs["goal_count"], distinct_goal_epis["mean"] ]
        header += ["reached_per_mean"]
        data += [reached_per_episode["mean"]]
        header += ["goal_entropy", "goal_value", "goal_policy_loss", "goal_value_loss", "goal_grad_norm"]
        data += [logs["goal_entropy"], logs["goal_value"], logs["goal_policy_loss"], logs["goal_value_loss"], logs["goal_grad_norm"]]


        txt_logger.info(
            "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f} R:μσmM {:.2f} {:.2f} {:.2f} {:.2f}|TR{:.4f} DI{} GCount {:.2f} DGCount {:.2f} ReachP {:.2f} | GH {:.3f} | GV {:.3f} | GpL {:.3f} | GvL {:.3f} | G∇ {:.3f}"
            .format(*data))

        if status["num_frames"] == 0:
            csv_logger.writerow(header)
        csv_logger.writerow(data)
        csv_file.flush()

        for field, value in zip(header, data):
            tb_writer.add_scalar(field, value, num_frames)

    # Save status

    if args.save_interval > 0 and update % args.save_interval == 0:
        status = {"num_frames": num_frames, "update": update,
                  "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict(), "goal_generator_state": goal_generator_model.state_dict(),
                  "goal_optimizer_state": algo.goal_optimizer.state_dict()}#"scheduler_state":algo.scheduler.state_dict(), "goal_scheduler_state":algo.generator_scheduler.state_dict()
        if hasattr(preprocess_obss, "vocab"):
            status["vocab"] = preprocess_obss.vocab.vocab
        utils_goal_multiple.save_status(status, model_dir)

        txt_logger.info("Status saved")
