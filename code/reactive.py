from multi_rendez import *
import numpy as np


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / 2 * np.power(sig, 2.))


class Satellite(object):

    def __init__(self, state, n):
        x0, y0, z0, dx0, dy0, dz0 = state

        self.state = np.array([x0, y0, z0, dx0, dy0, dz0])
        self.estimated_state = np.array([x0, y0, z0, dx0, dy0, dz0])
        self.n = n
        self.t = 0.

    def move_closer(self):
        ''' No path planning, just get closer in LVLH frame. '''
        x, y, z = self.estimated_state[0:3]
        target_y = 0.

        w = abs(y)
        sign = np.sign(target_y - y)
        T = np.array([0, sign, 0])

        return T, w

    def dont_hit(self):
        ''' Based off distance and relative velocity of target, fire away from target. '''
        x, y, z = self.estimated_state[0:3]
        target_y = 0.

        w = 12.88 * .975 ** abs(y)
        sign = np.sign(target_y - y)
        T = np.array([0., -sign, 0.])

        return T, w

    def station_keeping(self):
        ''' When target distance reached, hold position. '''
        target_y = 10.
        x, y, z = self.estimated_state[0:3]

        w = 25 * gaussian(y, target_y, 2)
        sign = np.sign(target_y - y)
        T = np.array([0, sign * self.state[4], 0])

        return T, w

    def stay_on_orbit(self):
        ''' If too far off orbital path in horizontal axis, push closer to axis. '''
        target_x = 0.
        x, y, z = self.estimated_state[0:3]

        w = abs(x)
        sign = np.sign(target_x - x)
        T = np.array([sign, 0., 0.])

        return T, w

    def act(self, t=1.):
        ''' Clohessy-Wiltshire equations. '''

        self.calculate_velocity()
        x0, y0, z0, dx0, dy0, dz0 = self.state
        n = self.n

        x = (4 - 3 * cos(n * t)) * x0 + (sin(n * t) / n) * dx0 + (2 / n) * (1 - cos(n * t)) * dy0
        y = 6 * (sin(n * t) - n * t) * x0 + y0 - (2 / n) * (1 - cos(n * t)) * dx0 + 1 / n * (4 * sin(n * t) - 3 * n * t) * dy0
        z = z0 * cos(n * t) + dz0 / n * sin(n * t)

        self.state = np.array([x, y, z, dx0, dy0, dz0])
        self.t += t

    def sense(self):
        ''' Use sensors to estimate relative location. '''

        x, y, z = self.state[0:3]

        # Some error in position estimation.
        err = 0.1
        x += random.uniform(-err, err)
        y += random.uniform(-err, err)
        z += random.uniform(-err, err)
        self.estimated_state[0:3] = np.array([x, y, z])

    def calculate_velocity(self):
        ''' Calculate thrusts and weights for each method, and the resulting output. '''

        # Get thrusts and weights
        T_1, w_1 = self.move_closer()
        T_2, w_2 = self.dont_hit()
        T_3, w_3 = self.station_keeping()
        T_4, w_4 = self.stay_on_orbit()

        Ts = [T_1, T_2, T_3, T_4]
        ws = [w_1, w_2, w_3, w_4]

        # Find weighted average thrust
        top, bottom = 0., 0.
        for T, w in zip(Ts, ws):
            top += T * w
            bottom += w

        # Update velocity state
        output = top / bottom
        vx, vy, vz = output

        # Jets have some error
        err = 0.01
        vx *= random.uniform(1 - err, 1 + err)
        vy *= random.uniform(1 - err, 1 + err)
        vz *= random.uniform(1 - err, 1 + err)
        output = np.array([vx, vy, vz])

        self.state[3:6] = output

n = 0.0011596575
initial_state = [5., 100., 0., 0., 0., 0.]
Inspector = Satellite(initial_state, n)

for _ in range(1000):
    print(Inspector.state[0:3])
    Inspector.sense()
    Inspector.act()
