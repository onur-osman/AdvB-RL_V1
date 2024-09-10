import time
from collections import deque, namedtuple

import gym
import numpy as np
import PIL.Image
import tensorflow as tf
import myutils as utils
import random
import keras.models as keras_models
import statistics

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam
import logging
tf.get_logger().setLevel(logging.ERROR)

def compute_loss(experiences, gamma, q_network, target_q_network):
    """
    Calculates the loss.

    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
      q_network: (tf.keras.Sequential) Keras model for predicting the q_values
      target_q_network: (tf.keras.Sequential) Keras model for predicting the targets

    Returns:
      loss: (TensorFlow Tensor(shape=(0,), dtype=int32)) the Mean-Squared Error between
            the y targets and the Q(s,a) values.
    """

    # Unpack the mini-batch of experience tuples
    states, actions, rewards, next_states, done_vals = experiences

    # Compute max Q^(s,a)
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)

    # Set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
    y_targets = rewards + (gamma * max_qsa * (1 - done_vals))

    # Get the q_values and reshape to match y_targets
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                tf.cast(actions, tf.int32)], axis=1))

    # Compute the loss
    loss = MSE(y_targets, q_values)

    return loss


@tf.function
def agent_learn(experiences, gamma):
    """
    Updates the weights of the Q networks.

    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.

    """

    # Calculate the loss
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network, target_q_network)

    # Get the gradients of the loss with respect to the weights.
    gradients = tape.gradient(loss, q_network.trainable_variables)

    # Update the weights of the q_network.
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

    # update the weights of target q_network
    utils.update_target_network(q_network, target_q_network)


def boltzman(_x, _t):
    _x_t = _x / _t
    e_x = np.exp(_x_t)
    _prob_act = e_x / (e_x.sum()+1e-12)

    _pp = np.random.rand(1)
    _sum_prob = 0
    for _k in range(len(_x[0])):
        _sum_prob += _prob_act[0][_k]
        if _pp <= _sum_prob:
            _action = _k
            break
    else:
        _action = np.random.randint(len(_x[0]))

    return _prob_act, _action

ENV_NAME = "CartPole-v1"

tf.random.set_seed(1234)

MEMORY_SIZE = 100000     # size of memory buffer
GAMMA = 0.995             # discount factor
ALPHA = 1e-3              # learning rate
NUM_STEPS_FOR_UPDATE = 1  # perform a learning update every C time steps
E_DECAY = 0.995
E_MIN = 0.01
env = gym.make(ENV_NAME)
stopping_criteria = 501
formation_period = 500
version_X = '01'

DQN_models = []
n_models = 10
discard_period = 20  # episode
save_models_period = 10
test_period = 1
n_best_model = 7
n_random_model = n_models - n_best_model

model_wins = []
with open('gym_CartPole_AdvB_10_weights.txt', 'r') as fp:
    for line in fp:
        line = line.strip()
        line = float(line)
        model_wins.append(line)
#model_wins = [100, 1, 0, 0, 0, 0, 0, 0, 0, 0]
for n_saves in range(n_models):
    if n_saves < 10:
        model_number = '0' + str(n_saves)
    else:
        model_number = str(n_saves)
    model_name = 'gym_CartPole_AdvB_10_' + model_number + '.h5'
    model_X = keras_models.load_model(model_name)
    DQN_models += [{'model': model_X, 'best_total_reward': 0, 'last_total_reward': 0, 'n_trials': 0, 'n_wins': model_wins[n_saves]}]


env.reset()
#PIL.Image.fromarray(env.render()[0])

state_size = env.observation_space.shape
num_actions = env.action_space.n

print('State Shape:', state_size)
print('Number of actions:', num_actions)

# Reset the environment and get the initial state.
current_state = env.reset()

# Select an action
action = 0

# Run a single time step of the environment's dynamics with the given action.
next_state, reward, done, _, _ = env.step(action)

# Display table with values.
#utils.display_table(current_state, action, next_state, reward, done)

# Replace the `current_state` with the state after the action is taken
current_state = next_state

total_reward_list = []
windowed_reward_list = []
total_reward_test_list = []
total_reward_adv_coop_list = []
dqn_performance_wins_list = []
dqn_performance_trials_list = []


start = time.time()

num_episodes = 100
max_num_timesteps = 1000

total_point_history = []
test_text_list = []
num_p_av = 100    # number of total points to use for averaging
epsilon = 1.0     # initial ε value for ε-greedy policy


for episode in range(num_episodes):

# TESTING
    if episode % test_period == 0:
        state = env.reset()[0]

        # Testing Advisory Board Cooperative Decision
        state_cooperative_decision = env.reset()[0]

        step = 0
        done = False
        total_points_dqns_coop = 0
        test_text_AdvB = 'not solved'
        while not done:
            prob_act_list = np.zeros((n_models, num_actions))
            prob_max = np.zeros((1, n_models))
            action_val_selected = np.zeros((1, num_actions))
            action_vals_cumulative = np.zeros((1, num_actions))
            action_member_list = []

            for k in range(n_models):
                model = DQN_models[k]['model']
                action_vals_swarm = model(state_cooperative_decision.reshape(1, -1))
                action_vals_cumulative += (DQN_models[k]['n_wins']) * np.array(action_vals_swarm)

            action = np.argmax(action_vals_cumulative)

            state_cooperative_decision, reward, terminate, truncate, _ = env.step(action)
            total_points_dqns_coop += reward

            if terminate or truncate:
                done = True

        print('Adv Cooperative Decision: {}'.format(total_points_dqns_coop))

        total_reward_adv_coop_list.append(total_points_dqns_coop)
        coopNet_averaged = np.mean(np.array(total_reward_adv_coop_list)[-100:])

print('Averaged Total Rewards {}'.format(coopNet_averaged))
#tot_time = time.time() - start
#print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")

# Plot the total point history along with the moving average
utils.plot_history(total_reward_adv_coop_list)
