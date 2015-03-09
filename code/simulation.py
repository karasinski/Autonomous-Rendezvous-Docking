from __future__ import division, print_function
from satellite import *
import dcomm as dc
import matplotlib
import pandas as pd
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import OrderedDict


def update_camera_position(inspector_position):
    ''' Convert from simulation to EDGE coordinate frame. '''
    cam_position = [0., 0., 0.]
    cam_position[0] = inspector_position[1]
    cam_position[1] = inspector_position[2]
    cam_position[2] = inspector_position[0]

    return cam_position


def RunSimulation(name, sensor):
    output = pd.Series()
    for initial_state in initial_conditions:
        print(initial_state)

        # Run the simulation many times
        number_of_simulations = 10
        trial = pd.Series()
        trial['x'] = initial_state[0]
        trial['y'] = initial_state[1]
        trial['z'] = initial_state[2]
        trial['type'] = name
        trial['sensor'] = sensor

        for _ in range(number_of_simulations):
            # Initialize Inspector
            if name is "Deliberative":
                Inspector = DeliberativeSatellite(initial_state, target_state, target, sensor)
            elif name is "Reactive":
                Inspector = ReactiveSatellite(initial_state, target_state, target, sensor)
            else:
                raise TypeError

            # Set the camera position to be equal to the Inspector position, sync
            # with server
            number_of_time_steps = 100000
            for time_step in range(number_of_time_steps):
                # Step the inspector
                Inspector.step()

                # Update the camera
                cam_position = update_camera_position(Inspector.state[0:3])
                CAM.position = cam_position
                dc.client()

                # Test if docked
                if Inspector.docked():
                    trial['time'] = Inspector.t
                    trial['fuel'] = Inspector.fuel
                    trial['distance'] = np.sum(np.abs(Inspector.state[1:3]))
                    trial['rate'] = np.sum(np.abs(Inspector.state[3:6]))
                    break

            if time_step == number_of_time_steps - 1:
                trial['time'] = np.nan
                trial['fuel'] = Inspector.fuel
                trial['distance'] = np.sum(np.abs(Inspector.state[1:3]))
                trial['rate'] = np.sum(np.abs(Inspector.state[3:6]))
            output = pd.concat((trial, output), axis=1)
            output.to_csv(name + ' ' + sensor)

    return output


def FullOutput(name, sensor, laser_error=0.1):
    for initial_state in initial_conditions:
        f = open(name + sensor + str(laser_error) + str(initial_state), 'w')
        print(initial_state)

        # Initialize Inspector
        if name is "Deliberative":
            Inspector = DeliberativeSatellite(initial_state, target_state, target, sensor, laser_error=laser_error)
        elif name is "Reactive":
            Inspector = ReactiveSatellite(initial_state, target_state, target, sensor, laser_error=laser_error)
        else:
            raise TypeError

        # Set the camera position to be equal to the Inspector position, sync
        # with server
        number_of_time_steps = 100000
        for time_step in range(number_of_time_steps):
            # Step the inspector
            Inspector.step()

            # Update the camera
            cam_position = update_camera_position(Inspector.state[0:3])
            CAM.position = cam_position
            dc.client()
            np.savetxt(f, np.concatenate([Inspector.state, Inspector.estimated_state]))

            # Test if docked
            if Inspector.docked():
                break

# Connect to EDGE
dc.connect()
dc.client()

# Find nodes
CAM = dc.Node("CM_Cam")
target = dc.Node("VR_PMA2_AXIAL_TARGET")

# Inspector initial and final targets
# initial_conditions = [[x, y, z, 0., 0., 0.] for x in range(100, 501, 200)
#                                             for y in range(-50,  51,  50)
#                                             for z in range(-37,  38,  37)]
initial_conditions = [[100.,  0.,  0., 0., 0., 0.],
                      [100., 50., 37., 0., 0., 0.],
                      [300.,  0.,  0., 0., 0., 0.],
                      [300., 50., 37., 0., 0., 0.],
                      [500.,  0.,  0., 0., 0., 0.],
                      [500., 50., 37., 0., 0., 0.]]

target_state = [5., 0., 0., 0., 0., 0.]

# Run 10x each initial condition for each sensor
# r_laser = RunSimulation("Reactive", "laser")
# d_laser = RunSimulation("Deliberative", "laser")

r_cv = RunSimulation("Reactive", "cv")
d_cv = RunSimulation("Deliberative", "cv")

# d = pd.concat((d_laser, d_cv, r_laser, r_cv), axis=1)
# d = d.T
# d.to_csv('output')

# Log all data for 1 trial of 1 initial condition for each sensor
# initial_conditions = [[100., 50., 37., 0., 0., 0.]]
# FullOutput("Reactive", "laser", laser_error=0.)
# FullOutput("Deliberative", "laser", laser_error=0.)

# FullOutput("Reactive", "laser")
# FullOutput("Deliberative", "laser")

# FullOutput("Reactive", "cv")
# FullOutput("Deliberative", "cv")

# l = np.loadtxt('Deliberativelaser[100, 0, 0, 0.0, 0.0, 0.0]')
# l = l.reshape(len(l)/12, 12)
