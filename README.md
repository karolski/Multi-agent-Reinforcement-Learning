# Multi-agent PPO
Implementation of proximal policy optimization(PPO) for multi-agent scenarios using tensorflow  

## environment
state space: continuous  
action space: discrete  

## dependencies
https://gitlab.com/karolski/muliti-agent-particle-env-fork

tensorflow 1.13
matplotlib

## Training
```bash
python train_<variant>.py
```
## Replay trained policy
```bash
python replay_<variant>.py
with setting a replay model location in the script

```
## Tensorboard
```
tensorboard --logdir=log
```
## LICENSE
MIT ICENSE

credits: