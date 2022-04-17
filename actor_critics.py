import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from collections import deque
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from Functional import softmax, log_softmax
import time
    
class ActorCritic:
    """T
        date : 19 / 04 / 21
    """
    def __init__(self, env, params):

        self.action_space = env.action_space
        self.state_space = env.state_space

        #set parameter for Keras-Tensorflow
        self.epsilon = params['epsilon'] 
        self.gamma = params['gamma'] 
        self.batch_size = params['batch_size'] 
        self.epsilon_min = params['epsilon_min'] 
        self.epsilon_decay = params['epsilon_decay'] 
        self.learning_rate = params['learning_rate']
        self.layer_sizes = params['layer_sizes']

        #higher maxlen -> resulting in higher score
        self.memory = deque(maxlen=2500)

        #better than iterative value
        self.model = self.build_model()


    def build_model(self):
        model = Sequential()
        for i in range(len(self.layer_sizes)):
            if i == 0:
                model.add(Dense(self.layer_sizes[i], input_shape=(self.state_space,), activation='relu'))
            else:
                model.add(Dense(self.layer_sizes[i], activation='relu'))
        model.add(Dense(self.action_space, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, imagine_state):
        values = self.model.predict(state)
        act_values = tuple(e1 * e2 for e1, e2 in zip(values[0], imagine_state))
        return np.argmax(act_values)

    def imagine(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        imagine_values = self.model.predict(state)
        return np.argmax(imagine_values[0])
        #return imagine_values[0]


    ''' 03 / 06 / 2021            
    def imagine(self, state):
        if np.random.rand() <= self.epsilon:
            imagine_values = self.model.predict(state)
        imagine_values = self.model.predict(state)
        return imagine_values[0]
    '''
    
    '''
    def act(self, state):
        #if np.random.rand() <= self.epsilon:
        #print("Random")
        #return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        #print ("Act Values: ",act_values)
        return np.argmax(act_values[0])
    '''
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
