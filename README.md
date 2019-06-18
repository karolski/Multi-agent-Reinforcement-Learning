# Multi-agent PPO
Implementation of proximal policy optimization(PPO) for multi-agent scenarios using tensorflow 

to be used with  https://gitlab.com/karolski/muliti-agent-particle-env-fork

## environment
- state space: continuous  
- action space: discrete  

## dependencies
- tensorflow 1.13
- matplotlib

## Training
To avoid cluttering master branch with logfiles and models create a new branch for each training run. This will also preserve hidden layer sizes.
```
python train_<variant>.py
```
## Replay trained policy
you need the same layers types and sizes in policy_net.py to replay the results successfully. 
```bash
python replay_<variant>.py
with setting a replay model location in the script

```
## Tensorboard
Preferably specify the folder of policy type in order to see the appropriate computational graph.
```
tensorboard --logdir=log/<name of logdir>
```
## LICENSE
MIT ICENSE

## Credits:
- this implementation is a modification of uidilr's [ppo_tf](https://github.com/uidilr/ppo_tf)
- single agent implementation is a modification of morvanZhou's [discrete_DPPO](ttps://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/12_Proximal_Policy_Optimization/discrete_DPPO.py
)
