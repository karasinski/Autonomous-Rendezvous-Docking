from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin
from numpy.linalg import solve
from cv_tools import *


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / 2 * np.power(sig, 2.))


class Satellite(object):

    ''' Our base class to hold non-architecture operations. '''

    def __init__(self, state, target_state, target, sensor):
        # Unpack input vectors
        x0, y0, z0, dx0, dy0, dz0 = state
        xf, yf, zf, dxf, dyf, dzf = target_state

        # Set basic parameters
        self.n = 0.0011596575
        self.t = 0.
        self.fuel = 0.
        self.laser_error = 0.1
        self.minimum_thrust = 0.01
        self.dt = 0.1
        self.name = 'Satellite'
        self.sensor = sensor

        # Set state, previous state, estimated state, and optimal velocity
        self.state = np.array([x0, y0, z0, dx0, dy0, dz0])
        self.previous_state = np.array([x0, y0, z0, dx0, dy0, dz0])
        self.estimated_state = np.array([x0, y0, z0, dx0, dy0, dz0])
        self.optimal_velocity = np.array([dx0, dy0, dz0])

        # Set final state
        self.target_state = np.array([xf, yf, zf, dxf, dyf, dzf])
        self.target = target

    def docked(self):
        ''' Define distance and velocity for successful capture '''

        close_enough = 0.5  # cm
        low_velocity = 0.5  # cm/s

        distance_offset = np.sum(np.abs(self.state[0:3] - self.target_state[0:3]))
        relative_velocity = np.sum(np.abs(self.state[3:6] - self.target_state[3:6]))

        # print(distance_offset, relative_velocity)
        return distance_offset < close_enough and relative_velocity < low_velocity

    def ClohessyWiltshire(self, t):
        '''Clohessy-Wiltshire equations'''

        n = self.n
        x0, y0, z0, dx0, dy0, dz0 = self.state

        x = (4 - 3 * cos(n * t)) * x0 + (sin(n * t) / n) * dx0 + (2 / n) * (1 - cos(n * t)) * dy0
        y = 6 * (sin(n * t) - n * t) * x0 + y0 - (2 / n) * (1 - cos(n * t)) * dx0 + 1 / n * (4 * sin(n * t) - 3 * n * t) * dy0
        z = z0 * cos(n * t) + dz0 / n * sin(n * t)

        self.state = np.array([x, y, z, dx0, dy0, dz0])
        self.t += t

    def sense(self):
        if self.sensor == 'laser':
            self.laser_sense()
        elif self.sensor == 'cv':
            self.cv_sense()
        else:
            raise TypeError

        # print(self.state[0:3], self.estimated_state[0:3])

    def laser_sense(self):
        ''' Use sensors to estimate relative location. '''

        x, y, z = self.state[0:3]

        # Some error in position estimation.
        x += np.random.uniform(-self.laser_error, self.laser_error)
        y += np.random.uniform(-self.laser_error, self.laser_error)
        z += np.random.uniform(-self.laser_error, self.laser_error)
        x *= np.random.uniform(1 - self.laser_error, 1 + self.laser_error)
        y *= np.random.uniform(1 - self.laser_error, 1 + self.laser_error)
        z *= np.random.uniform(1 - self.laser_error, 1 + self.laser_error)
        self.estimated_state[0:3] = np.array([x, y, z])

    def cv_sense(self):
        ''' Try to use CV or fallback to linear guess. '''

        try:
            # Scan the docking port
            image, contours = scan_docking_port(self.target)

            # Detect features
            center, distance = detect_features(image, contours)

            # Estimate state
            state = estimate_state(center, distance, image)
            # print('cv')

        except Exception:
            # State not found, guessing
            state = self.previous_state[0:3] + self.dt * self.previous_state[3:6]
            # print('guess')

        self.estimated_state[0:3] = np.array(state)

    def thrusters(self):
        ''' Try to fire thrusters at optimal rate, or do our best. '''

        # Jets have a minimum thrust output. Jets have some error, fire at +/-
        # a percentage of error.
        err = self.minimum_thrust / 10.

        if np.sum(np.abs(self.optimal_velocity)) > 1.:
            self.optimal_velocity = self.optimal_velocity/np.sum(np.abs(self.optimal_velocity))

        output = np.array([], dtype=np.float64)
        for rate, old_rate in zip(self.optimal_velocity, self.previous_state[3:6]):
            # If the desired rate is less than the minimum, don't burn.
            if abs(rate) < self.minimum_thrust:
                rate = np.random.uniform(-err, err)
            elif abs(rate - old_rate) < self.minimum_thrust:
                if abs(rate - old_rate) > self.minimum_thrust / 2:
                    sign = np.sign(rate - old_rate)
                    rate = old_rate + sign * self.minimum_thrust * np.random.uniform(1 - err, 1 + err)
                else:
                    rate = old_rate
            else:
                rate *= np.random.uniform(1 - err, 1 + err)

            output = np.append(output, rate)

        self.fuel += np.sum(np.abs(self.state[3:6] - output))
        self.state[3:6] = output

    def plot(self, results):
        '''Plot the position of the craft'''

        f, (ax1, ax2) = plt.subplots(2, 1)

        ax1.set_title('Position After Separation Using Multiple Burns')
        ax1.plot(results[:, 0], label='x')
        ax1.plot(results[:, 1], label='y')
        ax1.plot(results[:, 2], label='z')
        ax1.set_ylabel("Position")
        ax1.set_xlabel("Time (s)")
        ax1.legend(loc='best')

        ax2.set_title('Velocity After Separation Using Multiple Burns')
        ax2.plot(results[:, 3], label='dx')
        ax2.plot(results[:, 4], label='dy')
        ax2.plot(results[:, 5], label='dz')
        ax2.set_ylabel("Velocity")
        ax2.set_xlabel("Time (s)")
        ax2.legend(loc='best')

        plt.savefig(self.name + '.pdf')
        plt.close()


class ReactiveSatellite(Satellite):
    ''' A reactive satellite architecture. '''

    def __init__(self, state, target_state, target, sensor):
        # Additional properties
        Satellite.__init__(self, state, target_state, target, sensor)

        # Set type
        self.name = 'Reactive'

        # Set list of behaviors to use
        self.behaviors = [self.move_closer, self.dont_hit,
                          self.station_keeping, self.stay_on_orbit,
                          self.return_to_plane]

    def step(self):
        self.sense()
        self.act()

    def move_closer(self):
        ''' No path planning, just get closer in LVLH frame. '''

        x, y, z = self.estimated_state[0:3]
        target_x = self.target_state[0]

        w = abs(target_x - x) + 1
        sign = np.sign(target_x - x)
        T = np.array([sign, 0., 0.])

        return T, w

    def dont_hit(self):
        '''
        Based off distance and relative velocity of target, fire away from
        target.
        '''

        x, y, z = self.estimated_state[0:3]
        target_x = self.target_state[0]

        w = (39. / 40.) ** (x - target_x)
        sign = np.sign(target_x - x)
        T = np.array([-sign, 0., 0.])

        return T, w

    def station_keeping(self):
        '''
        When target distance reached: hold position, don't fire thrusters.
        '''

        x, y, z = self.estimated_state[0:3]
        target_x = self.target_state[0]

        w = 10. * gaussian(x, target_x, 2)
        T = np.array([0., 0., 0.])

        return T, w

    def stay_on_orbit(self):
        '''
        If too far off orbital path in horizontal axis, push closer to axis.
        '''

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

    def act(self):
        ''' Merge all behaviors and do a single burn for t seconds. '''

        self.calculate_velocity()
        self.thrusters()
        self.ClohessyWiltshire(self.dt)

    def calculate_velocity(self):
        '''
        Calculate thrusts and weights for each method, and the resulting
        output.
        '''

        # Set previous state to be current state, prepare to update current
        # state
        self.previous_state = self.state

        # Get thrusts and weights
        top, bottom = 0., 0.
        for behavior in self.behaviors:
            T, w = behavior()
            top += T * w
            bottom += w

        # Find weighted average thrust ('desired velocity')
        self.optimal_velocity = top / bottom


class DeliberativeSatellite(Satellite):

    ''' A deliberative satellite architecture. '''

    def __init__(self, state, target_state, target, sensor):
        # Call Satellite init method
        Satellite.__init__(self, state, target_state, target, sensor)

        # Set type
        self.name = 'Deliberative'

        # Additional properties
        self.burn_time = 60.

    def step(self):
        self.sense()
        self.plan()
        self.act()

    def plan(self):
        '''
        Determine parameters for orbital maneuver.
        Picks 1 of 3 traj. types: Homing, Closing Final Approach.
        '''
        x, y, z = self.estimated_state[0:3]
        distance = np.sqrt((x)**2 + (y)**2 + (z)**2)

        # Final approach
        if distance <= 50.0:
            self.burn_time = 0.2
        # Closing
        elif distance <= 500.0:
            self.burn_time = 1.0
        else:
            self.burn_time = 5.0

        # Compute the delta v
        self.DeltaV(self.burn_time)

    def act(self):
        '''
        Perform the rendezvous using open-loop thruster inputs.
        '''

        self.thrusters()

        t = 0.
        while t < self.burn_time:
            self.previous_state = self.state
            self.ClohessyWiltshire(self.dt)
            t += self.dt

    def DeltaV(self, t):
        '''
        Find the delta_v required to take the system to zero for some time.
        Assume you start with a known position (x0, y0, z0).
        '''

        x0, y0, z0, dx0, dy0, dz0 = self.state
        xf, yf, zf, dxf, dyf, dzf = self.target_state

        n = self.n

        # solve Ax = b
        b = np.array([(3. * cos(n * t) - 4.) * (x0 - xf), (y0 - yf) + 6. * (x0 - xf) * (sin(n * t) - n * t)])

        A = np.array([[(1. / n) * sin(n * t), (2. / n) * (1. - cos(n * t))],
                      [(2. / n) * (1. - cos(n * t)), 3. * t - (4. * sin(n * t) / n)]])

        x = solve(A, b)

        xd0 = x[0]
        yd0 = x[1]
        zd0 = -n * (z0 - zf) * cos(n * t) / sin(n * t)

        self.optimal_velocity = np.array([xd0, yd0, zd0])


def TestSatellite(Inspector):
    output = Inspector.state

    for _ in range(2000):
        old_state = np.array(Inspector.state)
        output = np.vstack((output, old_state))

        # Take a step
        Inspector.step()

        # Test if docked
        if Inspector.docked():
            break

    print(Inspector.name, "dV", Inspector.fuel, "t", Inspector.t)
    Inspector.plot(output)

# n = 0.0011596575
# initial_state = [100., 5., -5., 0., 0., 0.]
# target_state = [0., 0., 0., 0., 0., 0.]

# d = DeliberativeSatellite(initial_state, target_state, n)
# r = ReactiveSatellite(initial_state, target_state, n)

# TestSatellite(d)
# TestSatellite(r)
