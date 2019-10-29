'''
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
'''

import numpy as np
import math
import json
import matplotlib.pyplot as plt

#filename = 'full_scan_log'
filename = 'full_scan_reg'

results = np.load('data/' + filename + '_results.npy')
with open('data/' + filename + '_settings.txt') as f:
  settings = json.load(f)
gammas = settings['gammas']
widths = settings['widths']
num_sweeps = settings['num_sweeps']
window_size_sweeps = settings['window_size']
num_datapoints = settings['num_datapoints']
window_size_datapoints = int(math.floor(window_size_sweeps/num_sweeps*num_datapoints))

print(results.shape)
num_datapoints = results.shape[0]
num_gammas = len(gammas)
num_widths = len(widths)


font_size = 20
font_size_legend = 20
font_size_title = 20

plt.rc('font', size=font_size)  # controls default text sizes
plt.rc('axes', titlesize=font_size_title)  # fontsize of the axes title
plt.rc('axes', labelsize=font_size)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=font_size)  # fontsize of the tick labels
plt.rc('ytick', labelsize=font_size)  # fontsize of the tick labels
plt.rc('legend', fontsize=font_size_legend)  # legend fontsize
plt.rc('figure', titlesize=font_size)  # fontsize of the figure title

plt.figure()
for width_id in range(num_widths):
    mean_performance = np.mean(results[:window_size_datapoints, :, width_id], axis=0)
    plt.plot(gammas, mean_performance, linewidth=2.0, label='w: {}'.format(widths[width_id]))
plt.ylabel('performance')
plt.xlabel('$\gamma$')
plt.title('early performance')
plt.legend(loc='lower left')
plt.axis([0.1, 1.0, -0.1, 1.1])

plt.figure()
for width_id in range(num_widths):
    mean_performance = np.mean(results[num_datapoints - window_size_datapoints :, :, width_id], axis=0)
    plt.plot(gammas, mean_performance, linewidth=2.0, label='w: {}'.format(widths[width_id]))
plt.ylabel('performance')
plt.xlabel('$\gamma$')
plt.title('late performance')
plt.legend(loc='lower left')
plt.axis([0.1, 1.0, -0.1, 1.1])


# ###    Individual plot ##########
# plt.figure()
# width_id = 0
# gamma_id = 0
# eval_interval = num_sweeps // num_datapoints
# sweeps = [i*eval_interval for i in range(1,num_datapoints+1)]
# plt.plot(sweeps, results[:,gamma_id,width_id], linewidth=2.0, label='gamma: {}, w: {}'.format(gammas[gamma_id], widths[width_id]))
# plt.ylabel('performance')
# plt.xlabel('#sweeps')
# plt.legend(loc='lower right')

plt.show()


