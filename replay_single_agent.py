"""
replays what was saved in discrete_single_ppo.py
"""
import time
import numpy as np
import tensorflow as tf
from make_env import make_env


GAME = 'simple_port'
EP_LEN = 600
GRAPH_LOCATION = 'model/single_models/simple_port2019-06-16_19:56-200.meta'
CHECKPOINT_LOCATION = 'model/single_models'
env = make_env(GAME)
env.discrete_action_input = True
num_agents = env.n

sess = tf.Session()
saver = tf.train.import_meta_graph(GRAPH_LOCATION)
saver.restore(sess,tf.train.latest_checkpoint(CHECKPOINT_LOCATION))
graph = tf.get_default_graph()
all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

state = graph.get_tensor_by_name('state:-1')
pi=graph.get_tensor_by_name("pi/probs:-1")


def choose_actions(obs):
    prob_weights = sess.run(pi, feed_dict={state: obs[None, :]})[0]
    all_actions = []
    for i in range(num_agents):
        available_decisions = range(prob_weights.shape[-1])
        action = np.random.choice(available_decisions,
                                      p=prob_weights[i]) # select action w.r.t the actions prob
        all_actions.append(action)
    return all_actions

while True:
    all_obs = env.reset()
    for t in range(EP_LEN):
        env.render()
        all_actions = choose_actions(all_obs[0])
        all_obs, all_rews, dones, info = env.step(all_actions)
        time.sleep(0.03)
        if min(dones):
            break