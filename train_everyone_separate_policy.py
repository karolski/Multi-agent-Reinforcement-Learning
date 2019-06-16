#!/usr/bin/python3
import gym
import numpy as np
import tensorflow as tf
from policy_net import Policy_net
from ppo import PPOTrain
from datetime import datetime

from make_env import make_env

ITERATION = int(2e4)
GAMMA = 0.999

EPISODE_LEN = 600
ENV_NAME = 'simple_port'
timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")
logdir = './log/train_sep_policy/' + ENV_NAME + timestamp


def main():
    env = make_env(ENV_NAME)
    env.discrete_action_input = True
    env.seed(0)
    ob_space = env.observation_space[0]
    num_agents = env.n
    Policies =  [Policy_net('policy'+str(num), env, multi_agent=True) for num in range(num_agents)]
    Old_Policies = [Policy_net('old_policy'+str(num), env, multi_agent=True) for num in range(num_agents)]
    PPOs = [PPOTrain(Policy, Old_Policy, gamma=GAMMA, lr=1e-4) for Policy , Old_Policy in zip(Policies, Old_Policies)]
    saver = tf.train.Saver()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(logdir, sess.graph)
        sess.run(tf.global_variables_initializer())
        success_num = 0
        all_curr_rews = [0] * num_agents

        for iteration in range(ITERATION):  # episode
            all_observations = [ [] for _ in range(num_agents)]
            all_actions = [ [] for _ in range(num_agents)]
            all_v_preds = [ [] for _ in range(num_agents)]
            all_rewards = [ [] for _ in range(num_agents)]
            run_policy_steps = 0

            all_obs = env.reset()
            while True:  # run policy RUN_POLICY_STEPS which is much less than episode length
                run_policy_steps += 1
                all_acts = []
                for agent_id, obs in enumerate(all_obs):
                    obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                    act, v_pred = Policies[agent_id].act(obs=obs, stochastic=True)

                    v_pred = np.asscalar(v_pred)
                    act  = np.asscalar(act)
                    all_acts.append(act)

                    all_observations[agent_id].append(obs)
                    all_actions[agent_id].append(act)
                    all_v_preds[agent_id].append(v_pred)
                    all_rewards[agent_id].append(all_curr_rews[agent_id])

                next_all_obs, all_curr_rews, dones, info = env.step(all_acts)

                if min(dones) or run_policy_steps > EPISODE_LEN:
                    all_v_preds_next = [agent_v_preds[1:] + [0] for agent_v_preds in all_v_preds]  # next state of terminate state has 0 state value
                    all_curr_rews = [-10] * num_agents
                    break
                else:
                    all_obs = next_all_obs


            print("episode:", iteration, "rewards:", [sum(rs) for rs in all_rewards])
            writer.add_summary(
                tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=run_policy_steps)])
                , iteration)
            for agent_id in range(num_agents):
                rewards = all_rewards[agent_id]
                observations = all_observations[agent_id]
                v_preds = all_v_preds[agent_id]
                v_preds_next = all_v_preds_next[agent_id]
                actions = all_actions[agent_id]
                writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward_agent'+ str(agent_id),
                                                                      simple_value=sum(rewards))]) , iteration)

                gaes = PPOs[agent_id].get_gaes(rewards=rewards, v_preds=v_preds,
                                            v_preds_next=v_preds_next)

                # convert list to numpy array for feeding tf.placeholder
                observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
                actions = np.array(actions).astype(dtype=np.int32)
                rewards = np.array(rewards).astype(dtype=np.float32)
                v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
                gaes = np.array(gaes).astype(dtype=np.float32)
                gaes = (gaes - gaes.mean()) / gaes.std()

                PPOs[agent_id].assign_policy_parameters()

                inp = [observations, actions, rewards, v_preds_next, gaes]

                # train
                for epoch in range(4):
                    sample_indices = np.random.randint(low=0, high=observations.shape[0], size=64)  # indices are in [low, high)
                    sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                    PPOs[agent_id].train(obs=sampled_inp[0],
                              actions=sampled_inp[1],
                              rewards=sampled_inp[2],
                              v_preds_next=sampled_inp[3],
                              gaes=sampled_inp[4])

                summary = PPOs[agent_id].get_summary(obs=inp[0],
                                                  actions=inp[1],
                                                  rewards=inp[2],
                                                  v_preds_next=inp[3],
                                                  gaes=inp[4])[0]
                writer.add_summary(summary, iteration*num_agents+agent_id)

            if iteration % 2000 == 0:
                model_location = './model/model_sep/' + ENV_NAME + timestamp +iteration+ '.ckpt'
                saver.save(sess, model_location)
                print("model saved in ", model_location)

        writer.close()

        while True:
            all_obs = env.reset()
            for i in range(EPISODE_LEN):
                all_acts=[]
                for agent_id, obs in enumerate(all_obs):
                    obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                    act, v_pred = Policies[agent_id].act(obs=obs, stochastic=True)
                    act = np.asscalar(act)
                    all_acts.append(act)
                all_obs, all_rewards, dones, info = env.step(all_acts)
                env.render()
                if min(dones):
                    break

if __name__ == '__main__':
    main()
