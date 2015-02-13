from __future__ import division, print_function
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import cos, sin
from numpy.linalg import solve


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
        self.burn_time = 60.

        self.dt = 0.1

    def ClohessyWiltshire(self, t):
        '''Clohessy-Wiltshire equations'''

        n = self.n
        x0, y0, z0, dx0, dy0, dz0 = self.state  # self.previous_state

        x = (4 - 3 * cos(n * t)) * x0 + (sin(n * t) / n) * dx0 + (2 / n) * (1 - cos(n * t)) * dy0
        y = 6 * (sin(n * t) - n * t) * x0 + y0 - (2 / n) * (1 - cos(n * t)) * dx0 + 1 / n * (4 * sin(n * t) - 3 * n * t) * dy0
        z = z0 * cos(n * t) + dz0 / n * sin(n * t)

        self.state = np.array([x, y, z, dx0, dy0, dz0])

    def DeltaV(self, t):
        '''find the delta_v required to take the system to zero for some time.
        assume you start with a known position (x0, y0, z0) '''

        x0, y0, z0, dx0, dy0, dz0 = self.state
        n = self.n

        # solve Ax = b
        b = np.array([(3. * cos(n * t) - 4.) * x0, y0 + 6.0 * x0 * (sin(n * t) - n * t)])

        A = np.array([[(1. / n) * sin(n * t), (2. / n) * (1. - cos(n * t))],
                      [(2.0 / n) * (1.0 - cos(n * t)), 3.0 * t - (4.0 * sin(n * t) / n)]])

        x = solve(A, b)
        xd0 = x[0]
        yd0 = x[1]

        zd0 = -n * z0 * cos(n * t) / sin(n * t)

        # print("deltav: ", xd0, yd0, zd0)
        self.previous_state = np.array([x0, y0, z0, xd0, yd0, zd0])
        # self.state = np.array([x0, y0, z0, xd0, yd0, zd0])

    def ConductRendez(self):
        '''perform the rendezvous using the DeltaV values attained above
        and the ClohessyWiltshire equations to compute the new position'''
        t = 0

        t_burn_time = self.burn_time

        while t < t_burn_time:
            self.ClohessyWiltshire(self.dt)
            t += self.dt
            self.t = t

    def sense(self):
        ''' Use sensors to determine current state and build world model. '''

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

    def plan(self):
        '''Determine parameters for orbital maneuver.
        Picks 1 of 3 traj. types: Homing, Closing Final Approach.'''
        x, y, z = self.estimated_state[0:3]
        distance = math.sqrt((x)**2 + (y)**2 + (z)**2)

        # doesn't work if this line is omitted for 0.1 burn time
        # self.state[0:3] = self.estimated_state[0:3]

        # final approach
        if distance <= 500.0:
            self.burn_time = 0.2
        # closing
        elif distance <= 5000.0:
            self.burn_time = 1.0
        else:
            self.burn_time = 5.0

        # compute the delta v
        self.DeltaV(self.burn_time)

    def act(self):
        ''' Clohessy-Wiltshire equations. '''
        ''' Perform open-loop thruster inputs '''

        # check to see if the thrust value is reasonable
        minimum_thrust = 0.01
        error = minimum_thrust / 10.

        velocity = np.array([], dtype=np.float64)

        for rate, old_rate in zip(self.previous_state[3:6], self.state[3:6]):
            # Do not burn if the desired rate is less than the minimum
            if abs(rate) < minimum_thrust:
                rate = 0.
            elif abs(rate - old_rate) < minimum_thrust:
                if abs(rate - old_rate) > minimum_thrust / 2.:
                    sign = np.sign(rate - old_rate)
                    rate = old_rate + sign * minimum_thrust * \
                        np.random.uniform(1 - error, 1 + error)
                else:
                    rate = old_rate
            # else if the proposed thrust is large, give it a fraction of the
            # desired thrust
            else:
                rate *= minimum_thrust * np.random.uniform(1 - error, 1 + error)

            velocity = np.append(velocity, rate)

        self.state[3:6] = velocity

        # once a reasonable thrust value has been calculated, conduct
        # rendezvous
        self.ConductRendez()

    def Plot(self, results, dt, filename):
        '''Plot the position of the craft'''
        f, (ax1, ax2) = plt.subplots(2, 1)

        ax1.set_title('Position After Separation Using Multiple Burns')
        ax1.plot(results[:, 0], label='x')
        ax1.plot(results[:, 1], label='y')
        ax1.plot(results[:, 2], label='z')
        ax1.set_ylabel("Position")
        ax1.set_xlabel("Time (s)")
        ax1.legend()

        ax2.set_title('Velocity After Separation Using Multiple Burns')
        ax2.plot(results[:, 3], label='dx')
        ax2.plot(results[:, 4], label='dy')
        ax2.plot(results[:, 5], label='dz')
        ax2.set_ylabel("Velocity")
        ax2.set_xlabel("Time (s)")
        ax2.legend()

        plt.savefig(filename, format="png")
        plt.close()

n = 0.0011596575
initial_state = [100., 5., -5., 0., 0., 0.]
target_state = [0., 0., 0., 0., 0., 0.]

Inspector = Satellite(initial_state, target_state, n)

output = Inspector.state
fuel = 0.
close_enough = 0.05
low_velocity = 0.05

for _ in range(2000):
    old_state = np.array(Inspector.state)
    output = np.vstack((output, old_state))

    Inspector.sense()
    Inspector.plan()
    Inspector.act()
    print(Inspector.state)

    fuel += np.sum(np.abs(Inspector.state[3:6] - old_state[3:6]))

    # Test if docked
    distance_offset = Inspector.state[0:3] - Inspector.target_state[0:3]
    relative_velocity = Inspector.state[3:6] - Inspector.target_state[3:6]
    if all(distance_offset < close_enough) and all(relative_velocity < low_velocity):
        break


print("fuel_used", fuel)
Inspector.Plot(output, Inspector.dt, "inspect.png")
