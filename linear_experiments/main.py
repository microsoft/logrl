'''
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
'''

import numpy as np
import json
import time
from domain import Domain
from agent import Agent

# Settings  ##########################################################################

# Experiment settings
filename = 'default_reg'
num_sweeps = 110000
window_size = 10000  # number of sweeps that will be averaged over to get initial and final performance
num_datapoints = 1100
num_runs = 1
gammas = [0.1, 0.8, 0.85, 0.9, 0.94, 0.96, 0.98, 0.99]  ########
#gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.94, 0.96, 0.98, 0.99]  ########
widths = [2]
#widths = [1, 2, 3, 5] #############

# Domain settings
domain_settings = {}
domain_settings['stochasticity'] = 0.25
domain_settings['num_states'] = 50
domain_settings['min_reward'] = -1.0
domain_settings['max_reward'] = 1.0

# Agent settings
agent_settings = {}
agent_settings['log_mapping'] = False
agent_settings['initial_beta_reg'] = 1    # used if log_mapping = True
agent_settings['initial_beta_log'] = 1    # used if log_mapping = True
agent_settings['initial_alpha'] = 1       # used if log_mapping = False
agent_settings['final_beta_reg'] = 0.1
agent_settings['final_beta_log'] = 0.01
agent_settings['final_alpha'] = 0.001
agent_settings['decay_period'] = 10000   # number of sweeps over which step_size is annealed (from initial to final)
agent_settings['c'] = 1                 # used if log_mapping = True
agent_settings['h'] = 200               # used if log_mapping = True
agent_settings['Q_init_log'] = 0        # used if log_mapping = True
agent_settings['theta_init_reg'] = 0    # used if log_mapping = False
agent_settings['max_return'] = domain_settings['max_reward']
# max_return is used to bound the update-target (to improve stability)

#############################################################################################
if num_datapoints > num_sweeps:
    num_datapoints = num_sweeps
num_gammas = len(gammas)
num_widths = len(widths)
eval_interval = num_sweeps // num_datapoints
my_domain = Domain(domain_settings)
my_agent = Agent(agent_settings, my_domain)


start = time.time()
avg_performance = np.zeros([num_datapoints,num_gammas, num_widths])
for run in range(num_runs):
    for width_index in range(num_widths):
        width = widths[width_index]
        my_domain.set_tile_width(width)
        my_domain.init_representation()
        for gamma_index in range(num_gammas):
            gamma = gammas[gamma_index]
            print('***** run {}, width: {}, gamma: {} *****'.format(run +1, width, gamma))
            my_domain.set_gamma(gamma)
            my_agent.initialize()
            performance = np.zeros(num_datapoints)
            eval_no = 0
            for sweep in range(num_sweeps):
                my_agent.perform_update_sweep()
                if (sweep % eval_interval == 0) & (eval_no < num_datapoints):
                    performance[eval_no] = my_agent.evaluate()
                    eval_no += 1
            mean = np.mean(performance)
            print('mean = {}'.format(mean))
            alpha = 1/(run+1)
            avg_performance[:,gamma_index, width_index] = (1-alpha)*avg_performance[:,gamma_index, width_index] + alpha*performance
end = time.time()
print("time: {}s".format(end-start))

# Store results + some essential settings
settings = {}
settings['log_mapping'] = agent_settings['log_mapping']
settings['gammas'] = gammas
settings['widths'] = widths
settings['num_sweeps'] = num_sweeps
settings['window_size'] = window_size
settings['num_datapoints'] = num_datapoints
with open('data/' + filename + '_settings.txt', 'w') as json_file:
    json.dump(settings, json_file)
np.save('data/' +  filename + '_results.npy', avg_performance)

print('Done.')