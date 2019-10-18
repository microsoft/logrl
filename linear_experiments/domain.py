import numpy as np
import math

class Domain:
    # States  [-1] <- 0 <-> 1 <-> 2 <-> 3 <-> ... <-> n -1 -> [-1]   (-1 indicates terminal state)
    # Action 0: towards state 0
    # Action 1: toward state n-1
    # Left terminal state: max reward
    # Right terminal state: min reward


    def __init__(self, settings):
        self.gamma = None
        self.qstar = None
        self.num_states = settings['num_states']
        self.min_reward = settings['min_reward']
        self.max_reward = settings['max_reward']
        self.stoch = settings['stochasticity']
        return

    def _compute_num_features(self):
        self.num_tiles_per_tiling = math.ceil(self.num_states / self.tile_width) + 1
        num_features = self.num_tilings * self.num_tiles_per_tiling
        return num_features

    def get_num_features(self):
        return self.num_features

    def get_qstar(self):
        return self.qstar

    def set_tile_width(self, width):
        self.tile_width = width

    def get_num_states(self):
        return self.num_states

    def get_gamma(self):
        return self.gamma

    def set_gamma(self,gamma):
        self.gamma = gamma
        self.qstar = self._compute_qstar()

    def init_representation(self):
        self.num_tilings = self.tile_width
        self.feature_value = math.sqrt(1/self.num_tilings)     # value of active feature tilings-representation
        self.num_features = self._compute_num_features()
        return

    def take_action(self, state, action):
        assert(state >= 0)

        if np.random.random() < self.stoch:
            if action == 0:
                action = 1
            else:
                action = 0

        if (state == self.num_states-1) and (action == 1):
            next_state = -1
            reward = self.min_reward
        elif (state == 0) and (action == 0):
            next_state = -1
            reward = self.max_reward
        else:
            if action == 0:
                next_state = state-1
            else:
                next_state = state +1
            reward = 0
        return next_state, reward

    def get_features(self, state):

        if state < -1 or state >= self.num_states:
            print('state out-of-bounds!')
            assert(False)

        features = np.zeros(self.num_features)
        if state != -1:
            features = np.zeros(self.num_features)
            for t in range(self.num_tilings):
                tilde_id = math.floor((state + t)/self.tile_width)
                features[tilde_id + t * self.num_tiles_per_tiling] = self.feature_value
        return features

    def get_qstar_plus_min(self):
        q = np.zeros([self.num_states,2])
        v = np.zeros(self.num_states+2)
        v[0] = 0
        v[-1] = self.min_reward / self.gamma

        for i in range(10000):
            v[1:-1] = q[:,0]
            q[:, 0] = (1-self.stoch)*self.gamma*v[:-2] + self.stoch * self.gamma*v[2:]
            q[:, 1] = (1-self.stoch)*self.gamma*v[2:] + self.stoch * self.gamma*v[:-2]

        qstar_min = -1*q

        q = np.zeros([self.num_states,2])
        v = np.zeros(self.num_states+2)
        v[0] = self.max_reward / self.gamma
        v[-1] = 0

        for i in range(10000):
            v[1:-1] = q[:,0]
            q[:, 0] = (1-self.stoch)*self.gamma*v[:-2] + self.stoch * self.gamma*v[2:]
            q[:, 1] = (1-self.stoch)*self.gamma*v[2:] + self.stoch * self.gamma*v[:-2]

        qstar_plus = q

        return qstar_plus, qstar_min


    def _compute_qstar(self):
        q = np.zeros([self.num_states,2])
        v = np.zeros(self.num_states+2)
        v[0] = self.max_reward / self.gamma
        v[-1] = self.min_reward / self.gamma

        for i in range(10000):
            v[1:-1] = np.max(q, 1)
            q[:, 0] = (1-self.stoch)*self.gamma*v[:-2] + self.stoch * self.gamma*v[2:]
            q[:, 1] = (1-self.stoch)*self.gamma*v[2:] + self.stoch * self.gamma*v[:-2]

        return q

