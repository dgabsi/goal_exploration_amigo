#Goal-Exploration in sparse reward environemnts
### Daniela Stern- Gabsi 

### github- dgabsi/Goal-Exploration in sparse reward environemnts
(updates were made from danielaneuralx which is my working github)

Reinforcement learning has shown great success in training agents to operate independently in dense reward environments, 
but in sparse reward environments, agents struggle since there is not enough feedback that leads to improvement. 

AMIGo is a novel algorithm that was developed in 2021 in which two agents, a Teacher and a Student, 
learn to overcome the sparse reward problem by adversarial learning. 
The Teacher generates goals for the student to reach and be rewarded, while the Student permanence influences teacher reward. 
Both are also rewarded by the environment. Through acting adversarially, the agent learns to explore and succeed in the environment. 

This work proposes two improvements to AMIGo framework. 
AMIGo-Concurrent accelerates the learning process, and AMIGo-Multiple extends the framework to multiple goals. 
It tests the results on two challenging Mini-Grid environments and shows that while AMIGo-Concurrent did not outperform AMIGo, 
AMIGO-Multiple is superior in terms of speed of learning. This research hopes to contribute for more efficient learning in sparse reward environments. 

AMIGo paper:
Learning with AMIGo: Adversarially Motivated Intrinsic GOals
(Campero et al., 2015)(https://arxiv.org/abs/2006.12122))

The code if based on starter files of minigrid and on torch-ac framework.
Available at : https://github.com/lcswillems/torch-ac and https://github.com/lcswillems/rl-starter-files
The framework was updated and changed to add AMIGo capabilities and to Implement Amigo-Concurrent and Amigo-Multiple.

For reference I used AMIGO implementation AMIGo original code available at:
https://github.com/facebookresearch/adversarially-motivated-intrinsic-goals


To run experiments, please run the following:

For AMIGo experiment run:
- For FourRooms run:
  - python3 train_goal.py --env MiniGrid-FourRooms-v0 --frames 5000000 --procs 40 --lr 0.0002 --lr-teacher 0.0004 --generator_threshold -0.27 --fix_seed
- For DoorKey run:
  - python3 train_goal.py --env MiniGrid-DoorKey-8x8-v0  --frames 5000000 --procs 40 --lr 0.001 --lr-teacher 0.002 --generator_threshold -0.28 --with_amigo_nets


For AMIGo concurrent run:
- For FourRooms run:
  - python3 train_goal.py --env MiniGrid-FourRooms-v0 --train_together --frames 5000000 --procs 40 --lr 0.0002 --lr-teacher 0.0004 --generator_threshold -0.27 --fix_seed
- For DoorKey run:
  - python3 train_goal.py --env MiniGrid-DoorKey-8x8-v0  --train_together --frames 5000000 --procs 40 --lr 0.001 --lr-teacher 0.002 --generator_threshold -0.28 --with_amigo_nets


For AMIGo mutltiple run:
- For FourRooms run:
  - python3 train_goal_multiple.py --env MiniGrid-FourRooms-v0 --frames 5000000 --procs 40 --lr 0.0002 --lr-teacher 0.0004 --fix_seed  
- For DoorKey run:
  - python3 train_goal_multiple.py --env MiniGrid-DoorKey-8x8-v0  --frames 5000000 --procs 40 --lr 0.001 --lr-teacher 0.002 --with_amigo_nets 

For evaluate AMIGo concurrent run:
- For FourRooms run:
  - python3 evaluate_goal.py --env MiniGrid-FourRooms-v0 --model FOURROOMS_CONCURRENT 
- For DoorKey run:
  - python3 evaluate_goal.py --env MiniGrid-DoorKey-8x8-v0 --model DOORKEY_CONCURRENT --with_amigo-nets


For AMIGo mutltiple run:
- For FourRooms run:
  -python3 evaluate_goal_multiple.py --env MiniGrid-FourRooms-v0 --model FOURROOMS_MULTIPLE  
- For DoorKey run:
  - python3 evaluate_goal_multiple.py --env MiniGrid-DoorKey-8x8-v0 --model DOORKEY_MULTIPLE --with_amigo-nets

For AMIGo baseline run:
- For FourRooms run:
  - python3 -m scripts.train --algo ppo --env MiniGrid-FourRooms-v0 --frames 5000000 --fix_seed 
- For DoorKey run:
  - python3 -m scripts.train --algo ppo --env MiniGrid-DoorKey-8x8-v0 --frames 5000000 --fix_seed 

packages needed :
- torch 
- numpy
- pandas 
- gym-minigrid
- tensorboardX
- tensorboard
- pickle
- gym
