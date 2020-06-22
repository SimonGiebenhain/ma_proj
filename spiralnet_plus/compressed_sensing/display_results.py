import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
work_dir = osp.dirname(osp.realpath(__file__))
out_dir = osp.join(work_dir, 'out')

results = np.load(osp.join(out_dir, 'interpolation_exp', 'compressed_sensing_err.npy'))

fig, ax = plt.subplots()

x = results[0, :, 0]
mean_err = results[0, :, 1]
std_err = results[0, :, 2]
median_error = results[0, :, 3]
ax.plot(x, mean_err)
ax.plot(x, np.ones(mean_err.shape)*1.102)
#ax.plot(x, median_error)

#ax.fill_between(x, (mean_err-std_err), (mean_err+std_err), alpha=.1)

plt.show()