import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd


# Configure figures for production
WIDTH = 495.0  # the number latex spits out
FACTOR = 1.0   # the fraction of the width the figure should occupy
fig_width_pt = WIDTH * FACTOR

inches_per_pt = 1.0 / 72.27
golden_ratio = (np.sqrt(5) - 1.0) / 2.0      # because it looks good
fig_width_in = fig_width_pt * inches_per_pt  # figure width in inches
fig_height_in = fig_width_in * golden_ratio  # figure height in inches
fig_dims = [fig_width_in, fig_height_in]     # fig dims as a list

def load_data(file):
    d = np.loadtxt(file)
    d = d.reshape(len(d)/12, 12)
    d = pd.DataFrame(d)
    d.columns = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'xm', 'ym', 'zm', 9, 10, 11]
    d = d[['x', 'y', 'z', 'vx', 'vy', 'vz', 'xm', 'ym', 'zm']]
    d['xe'] = d.x - d.xm
    d['ye'] = d.y - d.ym
    d['ze'] = d.z - d.zm
    d = d * 2.54   # convert from inches to cm

    return d

def err_plot(laser1, laser2, cv1, cv2):
    plt.figure(figsize=fig_dims)
    plt.scatter(laser1.x, laser1.xe, color='r', label='Laser')
    plt.scatter(laser2.x, laser2.xe, color='r')
    plt.scatter(cv1.x, cv1.xe, color='b', label='CV')
    plt.scatter(cv2.x, cv2.xe, color='b')
    plt.legend(loc='best')
    plt.xlabel('x [cm]')
    plt.ylabel('Error [cm]')
    plt.xlim(min(min(laser1.x), min(cv1.x)) - 1, max(max(laser1.x), max(cv1.x)) + 1)
    plt.ylim(min(laser1.xe) - 1, -min(laser1.xe) + 1)
    plt.show()


def vel_plot(r, d, total=False):
    plt.figure(figsize=fig_dims)
    if total:
        plt.plot(d.x, abs(d.vx) + abs(d.vy) + abs(d.vz), color='r', label='Deliberative')
        plt.plot(r.x, abs(r.vx) + abs(r.vy) + abs(r.vz), color='b', label='Reactive')
        plt.ylabel('$v$ [cm/s]')
        plt.ylim(-0.05, 2.6)
    else:
        plt.plot(d.x, d.vx, color='r', label='Deliberative')
        plt.plot(r.x, r.vx, color='b', label='Reactive')
        plt.ylabel('$v_x$ [cm/s]')
        plt.ylim(min(r.vx) - 1, max(r.vx) + 1)
    plt.xlabel('x [cm]')
    plt.xlim(min(min(r.x), min(d.x)) - 1, max(max(r.x), max(d.x)) + 1)
    plt.legend(loc='lower right')
    plt.show()

dl = load_data('Deliberativelaser')
dc = load_data('Deliberativecv')
de = load_data('Deliberativeexact')
rl = load_data('Reactivelaser')
rc = load_data('Reactivecv')
re = load_data('Reactiveexact')

# err_plot(dl, rl, dc, rc)

# vel_plot(rc[1:], dc[1:], total=True)
# vel_plot(rc[1:], dc[1:])
# vel_plot(re[1:], de[1:], total=True)
# vel_plot(re[1:], de[1:])
