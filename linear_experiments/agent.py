import numpy as np

class Agent(object):

    def __init__(self, settings, domain):

        self.init_beta_reg = settings['initial_beta_reg']
        self.init_beta_log = settings['initial_beta_log']
        self.init_alpha = settings['initial_alpha']
        self.final_beta_reg = settings['final_beta_reg']
        self.final_beta_log = settings['final_beta_log']
        self.final_alpha = settings['final_alpha']
        self.decay_period = settings['decay_period']

        self.decay_beta_reg = (self.final_beta_reg/self.init_beta_reg)**(1./(self.decay_period-1))
        self.decay_beta_log = (self.final_beta_log/self.init_beta_log)**(1./(self.decay_period-1))
        self.decay_alpha = (self.final_alpha/self.init_alpha)**(1./(self.decay_period-1))

        self.theta_init_reg = settings['theta_init_reg']
        self.log_mapping = settings['log_mapping']
        self.c = settings['c']
        self.h = settings['h']
        self.Q_init_log = settings['Q_init_log']
        self.domain = domain
        self.num_states = domain.get_num_states()
        self.max_return = settings['max_return']


    def perform_update_sweep(self):

        num_samples = (self.num_states) * 2
        samples = [None] * num_samples

        i = 0
        for s in range(0, self.num_states):
            for a in [0, 1]:
                s2, r = self.domain.take_action(s, a)
                samples[i] = (s, a, s2, r)
                i += 1

        np.random.shuffle(samples)

        for sample in samples:
            self._single_update(sample)

        self.beta_log = max(self.beta_log*self.decay_beta_log, self.final_beta_log)
        self.beta_reg = max(self.beta_reg*self.decay_beta_reg, self.final_beta_reg)
        self.alpha = max(self.alpha*self.decay_alpha, self.final_alpha)
        return


    def initialize(self):
        self.num_features = self.domain.get_num_features()
        self.gamma = self.domain.get_gamma()
        self.qstar = self.domain.get_qstar()
        f = self.domain.get_features(0)

        self.alpha = self.init_alpha
        self.beta_reg = self.init_beta_reg
        self.beta_log = self.init_beta_log
        if self.log_mapping:
            self.d = -self.c*np.log(self.Q_init_log + self.gamma**self.h)
            self.theta_min = np.zeros([self.num_features, 2])
            self.theta_plus = np.zeros([self.num_features, 2])
            f = self.domain.get_features(0)
            v0_log = np.dot(self.theta_plus[:, 0], f) - np.dot(self.theta_min[:, 0], f)
            v0 = self._f_inverse(v0_log)
        else:
            self.theta = np.ones([self.num_features, 2]) * self.theta_init_reg
            f = self.domain.get_features(0)
            v0 = np.dot(self.theta[:, 0], f)
        print("reg. v(0): {:1.2f}".format(v0))

    def _single_update(self, sample):
        s, a, s2, r = sample
        f = self.domain.get_features(s)
        f2 = self.domain.get_features(s2)


        if self.log_mapping:
            if r >= 0:
                r_plus = r
                r_min = 0
            else:
                r_plus = 0
                r_min = -r

            #compute_optimal action
            q_next_0 = self._f_inverse(np.dot(self.theta_plus[:, 0], f2)) \
                      - self._f_inverse(np.dot(self.theta_min[:, 0], f2))
            q_next_1 = self._f_inverse(np.dot(self.theta_plus[:, 1], f2)) \
                      - self._f_inverse(np.dot(self.theta_min[:, 1], f2))
            if q_next_0 > q_next_1:
                a_star = 0
            else:
                a_star = 1


            # plus-network update
            if s2 == -1:  # terinal state
                v_next_log_plus = self._f(0.0)
            else:
                if a_star == 0:
                    v_next_log_plus = np.dot(self.theta_plus[:, 0], f2)
                else:
                    v_next_log_plus = np.dot(self.theta_plus[:, 1], f2)

            q_sa_log_plus = np.dot(self.theta_plus[:, a], f)
            q_sa_plus = self._f_inverse(q_sa_log_plus)
            v_next_plus = self._f_inverse(v_next_log_plus)
            update_target_plus = min(r_plus + self.gamma * v_next_plus, self.max_return)
            update_target_new_plus = q_sa_plus + self.beta_reg * (update_target_plus - q_sa_plus)
            TD_error_log_plus = self._f(update_target_new_plus) - q_sa_log_plus
            self.theta_plus[:, a] += self.beta_log * TD_error_log_plus * f

            # min-network update
            if s2 == -1:  # terinal state
                v_next_log_min = self._f(0.0)
            else:
                if a_star == 0:
                    v_next_log_min = np.dot(self.theta_min[:, 0], f2)
                else:
                    v_next_log_min = np.dot(self.theta_min[:, 1], f2)
            q_sa_log_min = np.dot(self.theta_min[:, a], f)
            q_sa_min = self._f_inverse(q_sa_log_min)
            v_next_min = self._f_inverse(v_next_log_min)
            update_target_min = min(r_min + self.gamma * v_next_min, self.max_return)
            update_target_new_min = q_sa_min + self.beta_reg * (update_target_min - q_sa_min)
            TD_error_log_min = self._f(update_target_new_min) - q_sa_log_min
            self.theta_min[:, a] += self.beta_log * TD_error_log_min * f

            if (self.theta_min > 100000).any():
                print('LARGE VALUE detected!')
            if np.isinf(self.theta_min).any():
                print('INF dectected!')
            elif np.isnan(self.theta_min).any():
                print('NAN dectected!')
            if (self.theta_plus > 100000).any():
                print('LARGE VALUE detected!')
            if np.isinf(self.theta_plus).any():
                print('INF dectected!')
            elif np.isnan(self.theta_plus).any():
                print('NAN dectected!')

        else:
            # compute update target
            if s2 == -1:  # terinal state
                v_next = 0.0
            else:
                q0_next = np.dot(self.theta[:, 0], f2)
                q1_next = np.dot(self.theta[:, 1], f2)
                v_next = max(q0_next, q1_next)
            q_sa = np.dot(self.theta[:, a], f)
            update_target = min(r + self.gamma * v_next, self.max_return)
            TD_error = update_target - q_sa
            self.theta[:, a] += self.alpha * TD_error * f

            if (self.theta > 100000).any():
                print('LARGE VALUE detected!')
            if np.isinf(self.theta).any():
                print('INF dectected!')
            elif np.isnan(self.theta).any():
                print('NAN dectected!')

    def evaluate(self):

        q = np.zeros([self.num_states,2])

        if self.log_mapping:
            for s in range(self.num_states):
                f = self.domain.get_features(s)
                q[s, 0] = self._f_inverse(np.dot(self.theta_plus[:, 0], f)) - self._f_inverse(np.dot(self.theta_min[:, 0], f))
                q[s, 1] = self._f_inverse(np.dot(self.theta_plus[:, 1], f)) - self._f_inverse(np.dot(self.theta_min[:, 1], f))
        else:
            for s in range(self.num_states):
                f = self.domain.get_features(s)
                q[s,0] = np.dot(self.theta[:, 0], f)
                q[s,1] = np.dot(self.theta[:, 1], f)

        i = np.argmax(q,axis=1)

        k = np.argmax(self.qstar)
        if (i == k).all():
            success = 1
        else:
            success = 0

        return success

    def _f(self, x):
        return self.c * np.log(x + self.gamma ** self.h) + self.d
        #return self.c * np.log(x)


    def _f_inverse(self,x):
        return np.exp((x - self.d )/ self.c) - self.gamma ** self.h
        #return np.exp(x/self.c)

