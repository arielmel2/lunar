
import random
import numpy as np
import gym
import tensorflow as tf
from gym import wrappers
import sys
from collections import deque
from keras import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras import backend as K
import matplotlib.pyplot as plt
import datetime

class Params:
    def __init__(self, epsilon = 1.0, epsilon_min = 0.1, epsilon_decay = 0.999, gamma= 0.99, alpha = 0.0002, 
                number_of_episodes =700, weights_file_name='lunar_weights.h5', mini_batch_size=256*32, 
                target_update_percentage=0.1, memory_len=10**6, target_weights_file_name='lunar_target_weights.h5'):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.alpha = alpha
        self.number_of_episodes = number_of_episodes
        self.weights_file_name = weights_file_name
        self.target_weights_file_name = target_weights_file_name
        self.mini_batch_size = mini_batch_size
        self.target_update_percentage = target_update_percentage
        self.memory_len = memory_len
        self.model = build_model()
        self.target_model = build_model() 
        
def get_sample(memory,  sample_batch_size):
    sample_batch = np.array(random.sample(memory, sample_batch_size))
    states       = np.array(list(sample_batch[:,0])).reshape(sample_batch_size, number_of_states)
    actions      = sample_batch[:,1].astype(int)
    rewards      = sample_batch[:,2]
    next_states  = np.array(list(sample_batch[:,3])).reshape(sample_batch_size, number_of_states)
    dones        = sample_batch[:,4].astype(bool)
    return states, actions, rewards, next_states, dones
    
def replay(params, memory):
    batch_size = min(params.mini_batch_size, len(memory))
    #print('replay')
    states, actions, rewards, next_states, dones = get_sample(memory, batch_size)
    loss = actual_replay(memory, params.model, params.target_model, params.gamma, batch_size, states, actions, rewards, next_states, dones) 
    loss = 0 if loss is None else loss
    update_target(params.model, params.target_model, params.target_update_percentage)
    return loss   

def update_target(model, target_model, target_update_percentage):
    if target_update_percentage >= 0.0:
        weights = model.get_weights()
        target_weights = target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = target_update_percentage * weights[i] + (1.0 - target_update_percentage) * target_weights[i]
        target_model.set_weights(target_weights)

def actual_replay(memory, model, target_model, gamma, batch_size, states, actions, rewards, next_states, dones):
    q = model.predict(states)
    target = target_model.predict(next_states)
    qtag = rewards + (1.0-dones) * np.max(gamma * target,axis=1)
    q[np.arange(batch_size),actions] = qtag
    #model.train_on_batch(s, expected)
    h = model.fit(states, q, epochs=1, verbose=0)
    return h.history['loss'][-1]


    predictions = np.amax(model.predict(next_states), axis=1)
    targets= rewards + (1-dones.astype(int)) * gamma * predictions
    #print('done', targets[dones], rewards[dones])
    target_f = model.predict(states)
    target_f[range(batch_size),np.reshape(actions, (1,batch_size))] = targets
    model.fit(states, target_f, epochs=1, verbose=0)

def train(env, params, number_of_states, number_of_actions):
    memory = deque(maxlen=params.memory_len)
    avg_n = 25
    last_n_rewards = np.zeros((100))
    sum_last_n_rewards = 0
    last_n_steps = np.zeros((100))
    sum_last_n_steps = 0
    
    total_n_reward = 0
    total_n_steps = 0
    episode_rewards_arr = []
    moving_avg_arr = []
    moving_steps_avg_arr = []


    for episode_num in range(1, params.number_of_episodes+1):
        episode_steps = 0
        episode_loss = 0
        episode_reward = 0
        episode_info_len = 0
        state = env.reset()
        #if (episode_num % 10 == 0):
        #    env.render()
        states = []
        next_states = []
        rewards = []
        dones = []
        actions = []
        
        done = False
        if episode_num < 100:
            random_first_altitude = random.random() + 0.5
            while state[1] > random_first_altitude:
                state,_,_,_ = env.step(0)
        state = np.reshape(state, [1, number_of_states])
        while not done:
            if params.epsilon > params.epsilon_min:
                params.epsilon *= params.epsilon_decay
            if random.random() < params.epsilon:
                action = random.randrange(number_of_actions)
            else:
                action = np.argmax(params.model.predict(state)[0])
            next_state, reward, done, _ = env.step(action)
            #if (episode_num % 10 == 0):
            #    env.render()
            next_state = np.reshape(next_state, [1, number_of_states])
            episode_steps += 1
            episode_reward += reward
            #print('epr=', episode_reward, 'r=',reward)
            times = 10 if reward > 99 else 1
            real_done = done if abs(reward) > 99 else False
            episode_info_len += times
            for i in range(times):
                states.append(state)
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(real_done)
                actions.append(action)
                #print('mem add:', state, action, reward, next_state, real_done)
                memory.append((state, action, reward, next_state, real_done))
            loss = 0
            if (episode_num % 32  == 0) or done:
                loss = replay(params, memory)
            episode_loss += loss 
            state = next_state
             
        #actual_replay(episode_info_len, np.array(states).reshape(episode_info_len, number_of_states), np.array(actions), np.array(rewards), 
        #                                                np.array(next_states).reshape(episode_info_len, number_of_states), np.array(dones))    
             
        
        episode_rewards_arr.append(episode_reward)
        sum_last_n_rewards = sum_last_n_rewards - last_n_rewards[episode_num % avg_n] + episode_reward
        sum_last_n_steps = sum_last_n_steps - last_n_steps[episode_num % avg_n] + episode_steps
        last_n_rewards[episode_num % avg_n] = episode_reward
        last_n_steps[episode_num % avg_n] = episode_steps
        if (episode_num > 100):
            moving_avg_arr.append(sum_last_n_rewards / avg_n)
            moving_steps_avg_arr.append(sum_last_n_steps / avg_n)
            if sum_last_n_rewards / avg_n > 125:
                params.epsilon = params.epsilon_min
            if sum_last_n_rewards / avg_n > 180:
                params.epsilon = params.epsilon_min / 2
        moving_average = round(sum_last_n_rewards / avg_n) if episode_num > avg_n else round(sum_last_n_rewards / episode_num)
        ploss = 100.0 * episode_loss / episode_steps
        print("Episode: {}, reward: {}, moving average: {}, steps: {}, epsilon: {}, loss: {}".format(episode_num, episode_reward, moving_average, episode_steps, round(params.epsilon,3), ploss))
       
        total_n_reward += episode_reward
        total_n_steps += episode_steps
        if (episode_num % avg_n == 0):
            print("=====> Episode batch: {}, reward: {}, steps: {}".format(episode_num / avg_n , 
                        round(total_n_reward / avg_n), round(total_n_steps / avg_n)))
            total_n_reward = 0 
            total_n_steps = 0 
            params.model.save_weights(params.weights_file_name)
            params.target_model.save_weights(params.target_weights_file_name)
            params.model.save_weights('m_' + str(episode_num) + '.h5')
            params.target_model.save_weights('tm_' + str(episode_num) + '.h5')
        sys.stdout.flush()
    return episode_rewards_arr, moving_avg_arr, moving_steps_avg_arr


def build_model():
    model = Sequential()
    model.add(Dense(128, input_dim=number_of_states, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(number_of_actions, activation='linear'))
    #model.compile(optimizer=Adam(lr=alpha), loss=huber_loss)
    model.compile(optimizer=Adam(lr=0.0002, decay=0.00001), loss=huber_loss)
    return model

def huber_loss(a, b):
    return tf.compat.v1.losses.huber_loss(a,b)

def play(times=10, render=True):
    print("play")
    #env = wrappers.Monitor(env, 'expt_dir', force=True)
    params = Params()
    params.model.load_weights(params.weights_file_name)
    total_rewards = 0
    total_steps = 0
    for t in range(times):
        print('starting', t)
        state = env.reset()
        state = np.reshape(state, [1, number_of_states])
        
        done = False
        episode_reward = 0
        episode_steps = 0
        while not done:
            if render:
                env.render()
            action = np.argmax(params.model.predict(state)[0])
            next_state, reward, done, _ = env.step(action)
            state = np.reshape(next_state, [1, number_of_states])
            episode_reward += reward 
            episode_steps += 1
        print(t, episode_steps, episode_reward)
        total_rewards += episode_reward
        total_steps += episode_steps
    print('summary: avg steps=', round(total_steps/times,1), 'avg reward=', round(total_rewards/times,1)) 


env = gym.make('LunarLander-v2')
number_of_states = env.observation_space.shape[0]
number_of_actions = env.action_space.n
if (len(sys.argv) > 1 and sys.argv[1] == "play"):
    play()
else:    
    print("learning")
    params = Params()
    rewards, moving_avg, moving_steps_avg = train(env, params, number_of_states, number_of_actions)

    print(len(rewards), len(moving_avg),len(moving_steps_avg))
    params.model.save_weights(params.weights_file_name)
    params.target_model.save_weights(params.target_weights_file_name)
    create_graph_training(rewards, moving_avg, moving_steps_avg,0)
