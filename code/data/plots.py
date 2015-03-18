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
    d['xe'] = d.x[:-1] - np.array(d.xm[1:])
    d['ye'] = d.y[:-1] - np.array(d.ym[1:])
    d['ze'] = d.z[:-1] - np.array(d.zm[1:])
    d['v'] = d[['vx', 'vy', 'vz']].abs().sum(axis=1)
    d = d * 2.54   # convert from inches to cm

    return d


def err_plot(e, s, k):
    f, ax = plt.subplots(figsize=fig_dims)
    e.plot(x='x', y='xe', ax=ax, color='b', style='.', markersize=2, label='Exact')
    s.plot(x='x', y='xe', ax=ax, color='r', style='.', markersize=2, label='Sensor')
    k.plot(x='x', y='xe', ax=ax, color='g', style='.', markersize=2, label='Kalman')
    plt.xlim(5 * 2.54, 500 * 2.54)
    handles, labels = ax.get_legend_handles_labels()
    labels = ['Exact', 'Sensor', 'Kalman']
    ax.legend(handles, labels, loc='best')
    plt.xlabel('x [cm]')
    plt.ylabel('Error [cm]')
    plt.grid(False)
    plt.show()


def vel_plot():
    f, ax = plt.subplots(figsize=fig_dims)
    re.plot(x='x', y='vx', ax=ax, color='b', style='.', markersize=2, label='Exact')
    rl.plot(x='x', y='vx', ax=ax, color='r', style='.', markersize=2, label='Sensor')
    rk.plot(x='x', y='vx', ax=ax, color='g', style='.', markersize=2, label='Kalman')
    plt.xlim(5 * 2.54, 500 * 2.54)
    plt.ylim(-2.6, .1)
    handles, labels = ax.get_legend_handles_labels()
    labels = ['Exact', 'Sensor', 'Kalman']
    ax.legend(handles, labels, loc='best')
    plt.xlabel('x [cm]')
    plt.ylabel('$v_x$ [cm/s]')
    plt.grid(False)
    plt.show()


def total_vel_plot():
    f, ax = plt.subplots(figsize=fig_dims)
    # re.plot(x='x', y='v', ax=ax, color='b', style='.', markersize=2, label='Exact')
    # rl.plot(x='x', y='v', ax=ax, color='r', style='.', markersize=2, label='Sensor')
    # rk.plot(x='x', y='v', ax=ax, color='g', style='.', markersize=2, label='Kalman')
    dck.plot(x='x', y='v', ax=ax, color='r', style='.', markersize=2, label='Deliberative')
    rck.plot(x='x', y='v', ax=ax, color='g', style='.', markersize=2, label='Reactive')
    plt.xlim(5 * 2.54, 250)
    # plt.ylim(-2.6, .1)
    handles, labels = ax.get_legend_handles_labels()
    labels = ['Deliberative', 'Reactive']
    ax.legend(handles, labels, loc='lower right')
    plt.xlabel('x [cm]')
    plt.ylabel('$v$ [cm/s]')
    plt.show()

# re = load_data('full_logs/Reactiveexact')
# rl = load_data('full_logs/Reactivelaser')
# rk = load_data('full_logs/ReactivelaserKalman')

# de = load_data('full_logs/Deliberativeexact')
# dck = load_data('full_logs/DeliberativecvKalman')
# rck = load_data('full_logs/ReactivecvKalman')

# err_plot(re, rl, rk)
# vel_plot()
# total_vel_plot()

# files = [
#  'Deliberative exact',
#  'Deliberative laser',
#  'Deliberative laser Kalman',
#  'Deliberative cv',
#  'Deliberative cv Kalman',
#  'Reactive exact',
#  'Reactive laser',
#  'Reactive laser Kalman',
#  'Reactive cv',
#  'Reactive cv Kalman']

def load(filename):
    df = pd.DataFrame.from_csv(filename)
    df = df.T.convert_objects(convert_numeric=True)
    df = df[['x', 'y', 'z', 'time', 'fuel', 'distance', 'rate']]
    df = df * 2.54
    df.time /= 2.54
    df.z = df.z.round().astype(int)
    return df

# for filename in files:
#     print(filename)
#     df = load(filename)
#     print(df.groupby(('x', 'y', 'z')).mean())
#     print('')

# df = load('Deliberative exact')
# d = df.groupby(('x', 'y', 'z')).mean()
# df = load('Reactive exact')
# r = df.groupby(('x', 'y', 'z')).mean()
# res = pd.concat((d, r), axis=1)
# print('Deliberative exact vs Reactive exact\n', res)

df = load('Deliberative laser')
dl = df.groupby(('x', 'y', 'z')).mean()
dl['sensor'] = 'Laser'
dl['type'] = 'Deliberative'
dl['filtered'] = False

df = load('Deliberative laser kalman')
dlk = df.groupby(('x', 'y', 'z')).mean()
dlk['sensor'] = 'Laser'
dlk['type'] = 'Deliberative'
dlk['filtered'] = True

df = load('Reactive laser')
rl = df.groupby(('x', 'y', 'z')).mean()
rl['sensor'] = 'Laser'
rl['type'] = 'Reactive'
rl['filtered'] = False

df = load('Reactive laser kalman')
rlk = df.groupby(('x', 'y', 'z')).mean()
rlk['sensor'] = 'Laser'
rlk['type'] = 'Reactive'
rlk['filtered'] = True

df = load('Deliberative cv')
dc = df.groupby(('x', 'y', 'z')).mean()
dc['sensor'] = 'CV'
dc['type'] = 'Deliberative'
dc['filtered'] = False

df = load('Deliberative cv kalman')
dck = df.groupby(('x', 'y', 'z')).mean()
dck['sensor'] = 'CV'
dck['type'] = 'Deliberative'
dck['filtered'] = True

df = load('Reactive cv')
rc = df.groupby(('x', 'y', 'z')).mean()
rc['sensor'] = 'CV'
rc['type'] = 'Reactive'
rc['filtered'] = False

df = load('Reactive cv kalman')
rck = df.groupby(('x', 'y', 'z')).mean()
rck['sensor'] = 'CV'
rck['type'] = 'Reactive'
rck['filtered'] = True

res = pd.concat((dl, dlk, rl, rlk, dc, dck, rc, rck))
res = res.query('x == 1270 and y == 127 and z == 94')
res.index = range(len(res))
print(res)


#             Deliberative     Reactive
#                time fuel   time  fuel
# x    y   z
# 254  0   0    94.6  2.55  104.5  4.95
#      127 94  181.7  2.55  191.7  9.74
# 762  0   0   294.8  2.56  304.6  5.01
#      127 94  381.6  2.61  391.7  9.64
# 1270 0   0   495.0  2.57  504.7  5.05
#      127 94  581.6  2.58  591.7  9.42

#                 Deliberative       Reactive
# sensor filtered   time  fuel   time    fuel
# Exact    False   581.6  2.58  591.7    9.42
# Laser    False   581.6  2.74  591.5  396.00
# Laser     True   581.6  2.59  590.0   50.54
#    CV    False   581.6  2.65  599.6  116.65
#    CV     True   581.6  2.60  592.7   26.44

# distances in cm, time in s, fuel in cm/s
