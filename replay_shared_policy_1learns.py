#!/usr/bin/python3
import time

import gym
import numpy as np
import tensorflow as tf
from policy_net import Policy_net
from ppo import PPOTrain
from datetime import datetime

from make_env import make_env

ITERATION = int(2e4)
GAMMA = 0.99
EPISODE_LEN = 600
ENV_NAME = 'simple_port'
GRAPH_PREFIX = 'model/ /simple_port2019-06-16_19:40.ckpt'

def main():
    timestamp = datetime.now().strftime("%H:%M:%S")
    env = make_env(ENV_NAME)
    env.discrete_action_input = True
    env.seed(0)
    ob_space = env.observation_space[0]
    num_agents = env.n
    Policy = Policy_net('policy', env, multi_agent=True)
    Old_Policy = Policy_net('old_policy', env, multi_agent=True)
    PPO = PPOTrain(Policy, Old_Policy, gamma=GAMMA, lr=2e-5)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./log/test_1learns_'+ENV_NAME+timestamp, sess.graph)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, GRAPH_PREFIX)
        all_obs = env.reset()
        # all_rewards = [0]*num_agents
        # success_num = 0
        #
        # for iteration in range(ITERATION):  # episode
        #     observations = []
        #     actions = []
        #     v_preds = []
        #     rewards = []
        #     run_policy_steps = 0
        #     while True:  # run policy RUN_POLICY_STEPS which is much less than episode length
        #         run_policy_steps += 1
        #         all_acts = []
        #         all_v_preds = []
        #         for obs in all_obs:
        #             obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
        #             act, v_pred = Policy.act(obs=obs, stochastic=True)
        #
        #             all_acts.append(np.asscalar(act))
        #             all_v_preds.append(np.asscalar(v_pred))
        #
        #         observations.append(all_obs[0])
        #         actions.append(all_acts[0])
        #         v_preds.append(all_v_preds[0])
        #         rewards.append(all_rewards[0])
        #
        #         next_all_obs, all_rewards, dones, info = env.step(all_acts)
        #
        #         if min(dones) or run_policy_steps > EPISODE_LEN:
        #             v_preds_next = v_preds[1:] + [0]  # next state of terminate state has 0 state value
        #             all_obs = env.reset()
        #             all_rewards = [-1]*num_agents
        #             break
        #         else:
        #             all_obs = next_all_obs
        #
        #     writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=run_policy_steps)])
        #                        , iteration)
        #     writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rewards))])
        #                        , iteration)
        #     print("episode:",str(iteration), "rewards:",sum(rewards))
        #
        #     gaes = PPO.get_gaes(rewards=rewards, v_preds=v_preds, v_preds_next=v_preds_next)
        #
        #     # convert list to numpy array for feeding tf.placeholder
        #     observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
        #     actions = np.array(actions).astype(dtype=np.int32)
        #     rewards = np.array(rewards).astype(dtype=np.float32)
        #     v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
        #     gaes = np.array(gaes).astype(dtype=np.float32)
        #     gaes = (gaes - gaes.mean()) / gaes.std()
        #
        #     PPO.assign_policy_parameters()
        #
        #     inp = [observations, actions, rewards, v_preds_next, gaes]
        #
        #     summary = PPO.get_summary(obs=inp[0],
        #                               actions=inp[1],
        #                               rewards=inp[2],
        #                               v_preds_next=inp[3],
        #                               gaes=inp[4])[0]
        #
        #     writer.add_summary(summary, iteration)
        # writer.close()

        while True:
            all_obs = env.reset()
            for i in range(EPISODE_LEN):
                all_acts=[]
                for obs in all_obs:
                    obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                    act, v_pred = Policy.act(obs=obs, stochastic=True)
                    act = np.asscalar(act)
                    all_acts.append(act)
                time.sleep(0.03)
                all_obs, all_rewards, dones, info = env.step(all_acts)
                env.render()
                if min(dones):
                    break

if __name__ == '__main__':
    main()
