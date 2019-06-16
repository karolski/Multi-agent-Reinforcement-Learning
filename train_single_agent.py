"""
A simple version of OpenAI's Proximal Policy Optimization (PPO). [https://arxiv.org/abs/1707.06347]
Distributing workers in parallel to collect data, then stop worker's roll-out and train PPO on collected data.
Restart workers once PPO is updated.
The global PPO updating rule is adopted from DeepMind's paper (DPPO):
Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
View more on my tutorial website: https://morvanzhou.github.io/tutorials
Dependencies:
tensorflow 1.8.0
gym 0.9.2


from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/12_Proximal_Policy_Optimization/discrete_DPPO.py

requires same agents
"""
import time
from datetime import datetime

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym, threading, queue
from make_env import make_env
N_WORKER = 12  # parallel workers

EP_MAX = 10
EP_LEN = 100
GAMMA = 0.999  # reward discount factor
A_LR = 0.0001  # learning rate for actor
C_LR = 0.0005  # learning rate for critic
MIN_BATCH_SIZE = 100  # minimum batch size for updating PPO
UPDATE_STEP = 15  # loop update operation n-steps
EPSILON = 0.2  # for clipping surrogate objective
HIDDEN_LAYER_SIZE = 60
GAME = 'simple_port'

env = make_env(GAME)
env.discrete_action_input = True
num_agents = env.n
S_DIM = env.observation_space[0].shape[0] #every agent sees all the others, so one observation is sufficient
A_DIM = env.action_space[0].n
A_DIM_SHAPE = [A_DIM]*num_agents
timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")

logdir = "log/single_agent/" + GAME + timestamp


class PPONet(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        w_init = tf.random_normal_initializer(0., .1)
        lc = tf.layers.dense(self.tfs, HIDDEN_LAYER_SIZE, tf.nn.relu, kernel_initializer=w_init, name='lc')
        self.v = tf.layers.dense(lc, 1)
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        tf.summary.histogram("reward", self.tfdc_r)
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        tf.summary.scalar("critic loss", self.closs)

        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        self.pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)

        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.int32, [None,num_agents], 'action')
        tf.summary.histogram("action dist", self.tfa)
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        tf.summary.histogram("advantage", self.tfadv)
        num_timesteps = tf.shape(self.tfa)[0]
        time_indexes = tf.range(num_timesteps)
        agent_indexes = tf.range(num_agents)
        self.t_idx = tf.stack([time_indexes]*num_agents, axis=1)
        self.agent_idx = tf.ones([num_timesteps, 1], tf.int32) * agent_indexes

        self.a_indices = tf.concat([
                        tf.expand_dims(self.t_idx, axis=-1),
                        tf.expand_dims(self.agent_idx, axis=-1),
                        tf.expand_dims(self.tfa, axis=-1),
                    ],
                    axis=-1)
        pi_prob = tf.gather_nd(params=self.pi, indices=self.a_indices)  # shape same as self.tfa
        oldpi_prob = tf.gather_nd(params=oldpi, indices=self.a_indices)  # shape=(None, )
        ratio = pi_prob / oldpi_prob
        surr = ratio * self.tfadv  # surrogate loss

        self.aloss = -tf.reduce_mean(tf.minimum(  # clipped surrogate objective
            surr,
            tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))
        tf.summary.scalar("actor loss", self.aloss)
        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        self.sess.run(tf.global_variables_initializer())
        self.summaries = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(logdir, self.sess.graph)


    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l_a = tf.layers.dense(self.tfs, HIDDEN_LAYER_SIZE, tf.nn.relu, trainable=trainable)
            output = tf.layers.dense(l_a, A_DIM*num_agents, tf.nn.relu, trainable=trainable)
            output_per_action = tf.reshape(output, [tf.shape(output)[0], num_agents, A_DIM])
            a_probs = tf.nn.softmax(output_per_action, name="probs") #apply softmax to each output row
            # tf.summary.histogram("probabilities", a_probs)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return a_probs, params

    def update(self):
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()  # wait until get batch of data
                self.sess.run(self.update_oldpi_op)  # copy pi to old pi
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]  # collect data from all workers
                data = np.vstack(data)
                s, a, r = data[:, :S_DIM], data[:, S_DIM: S_DIM + num_agents], data[:, -1:]
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
                # update actor and critic in a update loop
                for _ in range(UPDATE_STEP):
                    time_idx = self.sess.run(self.t_idx, {self.tfs: s, self.tfa: a, self.tfadv: adv})
                    agent_idx = self.sess.run(self.agent_idx, {self.tfs: s, self.tfa: a, self.tfadv: adv})
                    action_idx = self.sess.run(self.a_indices, {self.tfs: s, self.tfa: a, self.tfadv: adv})
                    self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv})

                summary = self.sess.run(self.summaries, {self.tfs: s, self.tfa: a,
                                                         self.tfadv: adv, self.tfdc_r: r})
                self.writer.add_summary(summary, GLOBAL_EP)
                for _ in range(UPDATE_STEP):
                    self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r})
                UPDATE_EVENT.clear()  # updating finished
                GLOBAL_UPDATE_COUNTER = 0  # reset counter
                ROLLING_EVENT.set()  # set roll-out available

    def choose_actions(self, s):  # run by a local
        prob_weights = self.sess.run(self.pi, feed_dict={self.tfs: s[None, :]})[0]
        if np.NaN in prob_weights[0]:
            print(prob_weights)
            prob_weights = self.sess.run(self.pi, feed_dict={self.tfs: s[None, :]})[0]
        all_actions = []
        for i in range(num_agents):
            available_decisions = range(prob_weights.shape[-1])
            action = np.random.choice(available_decisions,
                                  p=prob_weights[i]) # select action w.r.t the actions prob
            all_actions.append(action)
        return all_actions

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


class Worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.env = make_env(GAME)
        self.env.discrete_action_input = True
        self.ppo = GLOBAL_PPO

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            obs = self.env.reset()[0]
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():  # while global PPO is updating
                    ROLLING_EVENT.wait()  # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []  # clear history buffer, use new policy to collect data
                all_actions = self.ppo.choose_actions(obs)
                all_new_obs, all_rews, dones, _ = self.env.step(all_actions)
                if min(dones): all_rews = -10

                buffer_s.append(obs)
                buffer_a.append(all_actions)
                buffer_r.append(sum(all_rews))

                obs = all_new_obs[0]
                ep_r += sum(all_rews)

                GLOBAL_UPDATE_COUNTER += 1  # count to minimum batch size, no need to wait other workers
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE or min(dones):
                    if min(dones):
                        v_s_ = 0  # end of episode
                    else:
                        v_s_ = self.ppo.get_v(obs)

                    discounted_r = []  # compute discounted reward
                    for rew in buffer_r[::-1]:
                        v_s_ = rew + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, None]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))  # put data in the queue
                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()  # stop collecting data
                        UPDATE_EVENT.set()  # globalPPO update

                    if GLOBAL_EP >= EP_MAX:  # stop training
                        COORD.request_stop()
                        break

                    if min(dones): break

            # record reward changes, plot later
            if len(GLOBAL_RUNNING_R) == 0:
                GLOBAL_RUNNING_R.append(ep_r)
            else:
                GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1] * 0.9 + ep_r * 0.1)

            GLOBAL_EP += 1
            if GLOBAL_EP % 50 == 0 :
                saver.save(GLOBAL_PPO.sess, 'model/single_models/' + GAME + timestamp, global_step=GLOBAL_EP)
                print("model saved in ", 'model/single_models/single_ended' + GAME + timestamp + "-" + str(GLOBAL_EP))


            print('{0:.1f}%'.format(GLOBAL_EP / EP_MAX * 100), '|W%i' % self.wid, '|Ep_r: %.2f' % ep_r, )


if __name__ == '__main__':
    GLOBAL_PPO = PPONet()
    saver = tf.train.Saver()
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()  # not update now
    ROLLING_EVENT.set()  # start to roll out
    workers = [Worker(wid=i) for i in range(N_WORKER)]

    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_RUNNING_R = []
    COORD = tf.train.Coordinator()
    QUEUE = queue.Queue()  # workers putting data in this queue
    threads = []
    for worker in workers:  # worker threads
        t = threading.Thread(target=worker.work, args=())
        t.start()  # training
        threads.append(t)
    # add a PPO updating thread
    threads.append(threading.Thread(target=GLOBAL_PPO.update, ))
    threads[-1].start()
    COORD.join(threads)
    GLOBAL_PPO.writer.close()

    # plot reward change and test
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('Episode')
    plt.ylabel('Moving reward')
    plt.ion()
    plt.show()

    saver.save(GLOBAL_PPO.sess, 'model/single_models/' + GAME + timestamp, global_step=GLOBAL_EP)
    print("model saved in ", 'model/single_models/single_ended'+ GAME + timestamp + "-" + str(GLOBAL_EP))

    env = make_env(GAME)
    env.discrete_action_input = True
    while True:
        all_obs = env.reset()
        for t in range(EP_LEN):
            env.render()
            all_actions = GLOBAL_PPO.choose_actions(all_obs[0])
            all_obs, all_rews, dones, info = env.step(all_actions)
            time.sleep(0.03)
            if min(dones):
                break