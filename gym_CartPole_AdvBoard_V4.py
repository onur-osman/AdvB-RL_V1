import time
from collections import deque, namedtuple

import gym
import numpy as np
import PIL.Image
import tensorflow as tf
import myutils as utils
import random
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
formation_period = 400
version_X = '01'

DQN_models = []
n_models = 10
discard_period = 20  # episode
save_models_period = 10
test_period = 1
n_best_model = 7
n_random_model = n_models - n_best_model


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

# Create the Q-Network
q_network = Sequential([
    Input(state_size),
    Dense(units=32, activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=num_actions, activation='linear')
    ])

# Create the target Q^-Network
target_q_network = Sequential([
    Input(state_size),
    Dense(units=32, activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=num_actions, activation='linear')
    ])

optimizer = Adam(ALPHA)

# Create Advisory Board
for kkk in range(n_models):
    model_X = Sequential([
        Input(state_size),
        Dense(units=32, activation='relu'),
        Dense(units=32, activation='relu'),
        Dense(units=num_actions, activation='linear')
        ])
    DQN_models += [{'model': model_X, 'best_total_reward': -5000, 'last_total_reward': -5000, 'n_trials': 0, 'n_wins': 0}]

total_reward_list = []
windowed_reward_list = []
total_reward_test_list = []
total_reward_adv_coop_list = []
dqn_performance_wins_list = []
dqn_performance_trials_list = []

# Store experiences as named tuples
experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

start = time.time()

num_episodes = 1001
max_num_timesteps = 1000

total_point_history = []
test_text_list = []
num_p_av = 100    # number of total points to use for averaging
epsilon = 1.0     # initial ε value for ε-greedy policy

# Create a memory buffer D with capacity N
memory_buffer = deque(maxlen=MEMORY_SIZE)

# Set the target network weights equal to the Q-Network weights
target_q_network.set_weights(q_network.get_weights())

for episode in range(num_episodes):

    # Reset the environment to the initial state and get the initial state
    state = env.reset()[0]
    total_points = 0
    done = False
    test_text = 'not solved'

    for t in range(max_num_timesteps):

        # Chose action from the q_values of Advisory Board
        prob_act_list = np.zeros((n_models, num_actions))
        prob_max = np.zeros((1, n_models))
        action_val_selected = np.zeros((1, num_actions))
        action_vals_cumulative = np.zeros((1, num_actions))
        action_member_list = []

        if episode > formation_period:
            if random.random() > epsilon:
                for k in range(n_models):
                    model = DQN_models[k]['model']
                    action_vals_swarm = model(state.reshape(1, -1))
                    action_vals_cumulative += (DQN_models[k]['n_wins']) * np.array(action_vals_swarm)
                    
                action = np.argmax(action_vals_cumulative)
            else:
                action = random.choice(np.arange(num_actions))
        else:
            if random.random() > epsilon:
                q_values = q_network(state.reshape(1, -1))
                action = np.argmax(q_values)
            else:
                action = random.choice(np.arange(num_actions))

        #model_to_use = DQN_models[model_to_use_index]['model']
        #action_vals_AdvNet = model_to_use(state.reshape(1, -1))

        #action_val_selected[0] = prob_act_list[model_to_use_index, :]
        #action = utils.get_action(action_vals_AdvNet, epsilon)
        if random.random() > epsilon:
            action = action
        else:
            action = random.choice(np.arange(num_actions))


        #q_values = q_network(state.reshape(1, -1))
        #action = utils.get_action(q_values, epsilon)
        #probabilities, action = boltzman(action_vals_AdvNet, 200)

        # Take action A and receive reward R and the next state S'
        next_state, reward, terminate, truncate, _ = env.step(action)

        if terminate or truncate:
            done = True
        else:
            done = False

        # Store experience tuple (S,A,R,S') in the memory buffer.
        # We store the done variable as well for convenience.
        memory_buffer.append(experience(state, action, reward, next_state, done))

        # Only update the network every NUM_STEPS_FOR_UPDATE time steps.
        update = utils.check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)

        if update:
            # Sample random mini-batch of experience tuples (S,A,R,S') from D
            experiences = utils.get_experiences(memory_buffer)

            # Set the y targets, perform a gradient descent step,
            # and update the network weights.
            agent_learn(experiences, GAMMA)

        state = next_state.copy()
        total_points += reward

        if truncate:
            test_text = 'SOLVED'
            break
        if terminate:
            test_text = 'not solved'
            break

    test_text_list.append(test_text)
    total_point_history.append(total_points)
    av_latest_points = np.mean(total_point_history[-num_p_av:])

    # Update the ε value
    # epsilon = utils.get_new_eps(epsilon)
    epsilon = max(E_MIN, E_DECAY * epsilon)

    print(f"\rEpisode {episode+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}, Done: {test_text}", end="")

    # if (episode+1) % num_p_av == 0:
    #     print(f"\rEpisode {episode+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}, Done: {test_text}")

    # We will consider that the environment is solved if we get an
    # average of 200 points in the last 100 episodes.
    #if av_latest_points >= 200.0:
        #print(f"\n\nEnvironment solved in {episode+1} episodes!")
        #q_network.save('lunar_lander_model.h5')
        #break


# TESTING
    if episode % test_period == 0:
        state = env.reset()[0]
        total_points_TrainedNetwork = 0
        test_text = 'not solved'

        for t in range(max_num_timesteps):

            # From the current state S choose an action A using an ε-greedy policy
            q_values = q_network(state.reshape(1,-1))
            action = np.argmax(q_values)
            # Take action A and receive reward R and the next state S'
            state, reward, terminate, truncate, _ = env.step(action)
            total_points_TrainedNetwork += reward
            if truncate:
                test_text = 'SOLVED'
                break
            if terminate:
                test_text = 'not solved'
                break


        # Testing Advisory Board Members
        solved_list = []
        for number, my_dqn_dic in enumerate(DQN_models):
            my_model = my_dqn_dic['model']
            my_dqn_dic['n_trials'] += 1
            my_n_wins = my_dqn_dic['n_wins']

            state = env.reset()[0]

            step = 0
            done = False
            total_points_dqns = 0
            test_text_AdvB = 'not solved'

            for t in range(max_num_timesteps):
                # From the current state S choose an action A using an ε-greedy policy
                q_values = my_model(state.reshape(1, -1))
                action = np.argmax(q_values)
                # Take action A and receive reward R and the next state S'
                state, reward, terminate, truncate, _ = env.step(action)
                total_points_dqns += reward
                if truncate:
                    test_text_AdvB = 'SOLVED'
                    break
                if terminate:
                    test_text_AdvB = 'not solved'
                    break

            solved_list.append(test_text_AdvB)
            DQN_models[number]['last_total_reward'] = total_points_dqns

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


        # Evaluation
        evaluation_list = []
        for nn_models in range(n_models):
            evaluation_list.append(DQN_models[nn_models]['last_total_reward'])
        best_model_index = np.argmax(np.array(evaluation_list))
        if episode > formation_period:
            DQN_models[best_model_index]['n_wins'] += 1
        print('Training Network Reward: {:.2f}, Advisory Board Reward: {:.2f}, Adv Cooperative Decision: {}'
              .format(total_points_TrainedNetwork, DQN_models[best_model_index]['last_total_reward'], total_points_dqns_coop))
        #print('***********************************************************')

        total_reward_adv_coop_list.append(total_points_dqns_coop)
        total_reward_list.append(DQN_models[best_model_index]['last_total_reward'])
        total_reward_test_list.append(total_points_TrainedNetwork)
        coopNet_averaged = np.mean(np.array(total_reward_adv_coop_list)[-100:])
        #if len(total_reward_list) >= window_size:
            #windowed_reward = np.sum(total_reward_list[-10:]) / window_size
            #windowed_reward_list.append(windowed_reward)
            # visualize_loss(windowed_reward_list)
        try:
            with open(r'gym_CartPole_AdvBTraining_AdvBoard_0995_10Adv_'+version_X+'.txt', 'w') as fp:
                fp.write('\n'.join(str(item) for item in total_reward_list))
            with open(r'gym_CartPole_AdvBTraining_TrainingNet_0995_10Adv_'+version_X+'.txt', 'w') as fp:
                fp.write('\n'.join(str(item) for item in total_reward_test_list))
            with open(r'gym_CartPole_AdvBTraining_Cooperative_0995_10Adv_'+version_X+'.txt', 'w') as fp:
               fp.write('\n'.join(str(item) for item in total_reward_adv_coop_list))
        except:
            print('could not reach internet')

        if episode % save_models_period == 0:
            for n_saves in range(n_models):
                if n_saves < 10:
                    model_number = '0' + str(n_saves)
                else:
                    model_number = str(n_saves)
                model_name = 'gym_CartPole_AdvB_10_' + model_number + '.h5'
                model_to_save = DQN_models[n_saves]['model']
                model_to_save.save(model_name)


            q_network.save('gym_CartPole_AdvB10_q_network.h5')
            # model_target.save('VRP_DQN_model_target_MA_5city_2agent_02.h5')

        # Discard in Best Models
        for nn_best_models in range(n_best_model):
            if total_points_TrainedNetwork > DQN_models[nn_best_models]['best_total_reward']:
                model_X = Sequential([
                    Input(state_size),
                    Dense(units=32, activation='relu'),
                    Dense(units=32, activation='relu'),
                    Dense(units=num_actions, activation='linear')
                ])
                New_model = {'model': model_X, 'best_total_reward': total_points_TrainedNetwork,
                             'last_total_reward': total_points_TrainedNetwork, 'n_trials': 1, 'n_wins': 1}
                DQN_models.insert(nn_best_models, New_model)
                DQN_models.pop(n_best_model)
                # model_best = tf.keras.Model(inputs=input_layer, outputs=output_layer)
                model_best = DQN_models[nn_best_models]['model']
                m1_weights = q_network.get_weights()
                model_best.set_weights(m1_weights)
                break

        # Discard worst model in randomly selected models
        if n_random_model>0 and episode % discard_period == 0:
            dqn_performance = []
            for nn_random_models in range(n_best_model, n_models):
                dqn_performance.append(
                    DQN_models[nn_random_models]['n_wins'] / DQN_models[nn_random_models]['n_trials'])
            worst_model_index = np.argmin(np.array(dqn_performance))
            model_random = DQN_models[n_best_model + worst_model_index]['model']
            m1_weights = q_network.get_weights()
            model_random.set_weights(m1_weights)
            DQN_models[n_best_model + worst_model_index]['best_total_reward'] = 0
            DQN_models[n_best_model + worst_model_index]['last_total_reward'] = total_points_TrainedNetwork
            DQN_models[n_best_model + worst_model_index]['n_trials'] = 0
            DQN_models[n_best_model + worst_model_index]['n_wins'] = 0

        dqn_performance = []
        dqn_performance_wins = []
        dqn_performance_trials = []
        for nn_random_models in range(n_models):
            dqn_performance_wins.append(DQN_models[nn_random_models]['n_wins'])
            dqn_performance_trials.append(DQN_models[nn_random_models]['n_trials'])

        #if len(total_reward_list) >= window_size:
        dqn_performance_wins_list.append(dqn_performance_wins)
        dqn_performance_trials_list.append(dqn_performance_trials)
        # visualize_loss(windowed_reward_list)
        try:
            if episode % 10 == 0:
                with open(r'gym_CartPole_AdvBoard_wins.txt', 'w') as fp:
                    fp.write('\n'.join(str(item) for item in dqn_performance_wins_list))
                with open(r'gym_CartPole_AdvBoard_trials.txt', 'w') as fp:
                    fp.write('\n'.join(str(item) for item in dqn_performance_trials_list))
        except:
            print('could not reach internet')

        # stopping criterion
        if coopNet_averaged >= stopping_criteria:
            print('stopping criteria is met')

            try:
                with open(r'gym_CartPole_AdvBTraining_AdvBoard_0995_10Adv_'+version_X+'.txt', 'w') as fp:
                    fp.write('\n'.join(str(item) for item in total_reward_list))
                with open(r'gym_CartPole_AdvBTraining_TrainingNet_0995_10Adv_'+version_X+'.txt', 'w') as fp:
                    fp.write('\n'.join(str(item) for item in total_reward_test_list))
                with open(r'gym_CartPole_AdvBTraining_Cooperative_0995_10Adv_'+version_X+'.txt', 'w') as fp:
                    fp.write('\n'.join(str(item) for item in total_reward_adv_coop_list))
            except:
                print('something went wrong')

            try:
                with open(r'gym_CartPole_AdvBoard_wins.txt', 'w') as fp:
                    fp.write('\n'.join(str(item) for item in dqn_performance_wins_list))
                with open(r'gym_CartPole_AdvBoard_trials.txt', 'w') as fp:
                    fp.write('\n'.join(str(item) for item in dqn_performance_trials_list))
            except:
                print('could not reach internet')

            for n_saves in range(n_models):
                if n_saves < 10:
                    model_number = '0' + str(n_saves)
                else:
                    model_number = str(n_saves)
                model_name = 'gym_CartPole_AdvB_10_' + model_number + '.h5'
                model_to_save = DQN_models[n_saves]['model']
                model_to_save.save(model_name)


            # model_target.save('VRP_DQN_model_target_MA_5city_2agent_02.h5')

            break

q_network.save('gym_CartPole_AdvB10_q_network.h5')
with open(r'gym_CartPole_AdvB_10_weights.txt', 'w') as fp:
    fp.write('\n'.join(str(item) for item in dqn_performance_wins_list[-1]))

tot_time = time.time() - start
print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")

# Plot the total point history along with the moving average
utils.plot_history(total_point_history)
