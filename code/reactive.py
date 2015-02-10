from __future__ import division, print_function
import numpy as np
from numpy import cos, sin


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / 2 * np.power(sig, 2.))


class Satellite(object):
    ''' Our 'satellite' class to hold all behaviours and sense/act. '''

    def __init__(self, state, target, n):
        # Unpack input vectors
        x0, y0, z0, dx0, dy0, dz0 = state
        xf, yf, zf, dxf, dyf, dzf = target

        # Set basic parameters
        self.n = n
        self.t = 0.

        # Set state, previous state, and estimated state
        self.state = np.array([x0, y0, z0, dx0, dy0, dz0])
        self.previous_state = np.array([x0, y0, z0, dx0, dy0, dz0])
        self.estimated_state = np.array([x0, y0, z0, dx0, dy0, dz0])

        # Set final state
        self.target_state = np.array([xf, yf, zf, dxf, dyf, dzf])

        # Set list of behaviors to use
        self.behaviors = [self.move_closer, self.dont_hit,
                          self.station_keeping, self.stay_on_orbit,
                          self.return_to_plane]

    def move_closer(self):
        ''' No path planning, just get closer in LVLH frame. '''

        x, y, z = self.estimated_state[0:3]
        target_x = self.target_state[0]

        w = abs(target_x - x) + 1
        sign = np.sign(target_x - x)
        T = np.array([sign, 0., 0.])

        return T, w

    def dont_hit(self):
        ''' Based off distance and relative velocity of target, fire away from target. '''

        x, y, z = self.estimated_state[0:3]
        target_x = self.target_state[0]

        w = (39. / 40.) ** (x - target_x)
        sign = np.sign(target_x - x)
        T = np.array([-sign, 0., 0.])

        return T, w

    def station_keeping(self):
        ''' When target distance reached: hold position, don't fire thrusters. '''

        x, y, z = self.estimated_state[0:3]
        target_x = self.target_state[0]

        w = 25. * gaussian(x, target_x, 2)
        T = np.array([0., 0., 0.])

        return T, w

    def stay_on_orbit(self):
        ''' If too far off orbital path in horizontal axis, push closer to axis. '''

        x, y, z = self.estimated_state[0:3]
        target_y = self.target_state[1]

        w = (target_y - y) ** 2 + abs(target_y - y)
        sign = np.sign(target_y - y)
        T = np.array([0., sign, 0.])

        return T, w

    def return_to_plane(self):
        ''' If too far out of plane, return. '''

        x, y, z = self.estimated_state[0:3]
        target_z = self.target_state[2]

        w = (target_z - z) ** 2 + abs(target_z - z)
        sign = np.sign(target_z - z)
        T = np.array([0., 0., sign])

        return T, w

    def act(self, t=0.1):
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
        x += np.random.uniform(-err, err)
        y += np.random.uniform(-err, err)
        z += np.random.uniform(-err, err)
        x *= np.random.uniform(1 - err, 1 + err)
        y *= np.random.uniform(1 - err, 1 + err)
        z *= np.random.uniform(1 - err, 1 + err)
        self.estimated_state[0:3] = np.array([x, y, z])

    def calculate_velocity(self):
        ''' Calculate thrusts and weights for each method, and the resulting output. '''

        # Set previous state to be current state, prepare to update current state
        self.previous_state = self.state

        # Get thrusts and weights
        top, bottom = 0., 0.
        for behavior in self.behaviors:
            T, w = behavior()
            top += T * w
            bottom += w

        # Find weighted average thrust ('desired velocity')
        optimal_output = top / bottom
        vx, vy, vz = optimal_output
        vx_old, vy_old, vz_old = self.previous_state[3:6]

        # Jets have a minimum thrust output. Jets have some error, fire at +/-
        # a percentage of error.
        minimum_thrust = 0.01
        err = minimum_thrust / 10.

        output = np.array([], dtype=np.float64)
        for rate, old_rate in zip(optimal_output, self.previous_state[3:6]):
            # If the desired rate is less than the minimum, don't burn.
            if abs(rate) < minimum_thrust:
                rate = 0.
            elif abs(rate - old_rate) < minimum_thrust:
                if abs(rate - old_rate) > minimum_thrust / 2:
                    sign = np.sign(rate - old_rate)
                    rate = old_rate + sign * minimum_thrust * np.random.uniform(1 - err, 1 + err)
                else:
                    rate = old_rate
            else:
                rate *= np.random.uniform(1 - err, 1 + err)

            output = np.append(output, rate)

        self.state[3:6] = output

# n = 0.0011596575
# initial_state = [100., 5., -5., 0., 0., 0.]
# target_state = [0., 0., 0., 0., 0., 0.]
# Inspector = Satellite(initial_state, target_state, n)

# for _ in range(2000):
#     print(Inspector.state[0:3])
#     Inspector.sense()
#     Inspector.act()
