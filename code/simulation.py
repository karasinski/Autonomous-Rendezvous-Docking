from reactive import *
import dcomm as dc
from time import sleep
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


# Connect to EDGE
dc.connect()
dc.client()

# Find camera node
CAM = dc.Node("CM_Cam")

# Simulation parameters
n = 0.0011596575
close_enough = 0.05
low_velocity = 0.05

# Inspector initial and final parameters
initial_conditions = [[x, y, z, 0., 0., 0.] for x in range(100, 501, 100)
                                            for y in range(-50,  51,  50)
                                            for z in range(-50,  51,  50)]
target_state = [5., 0., 0., 0., 0., 0.]

output = OrderedDict()
for initial_state in initial_conditions:
    print(initial_state)

    # Run the simulation many times
    number_of_simulations = 100
    trial_output = []
    for _ in range(number_of_simulations):
        # Initialize Inspector
        Inspector = Satellite(initial_state, target_state, n)

        # Set the camera position to be equal to the Inspector position, sync
        # with server
        number_of_time_steps = 100000
        for time_step in range(number_of_time_steps):
            # Step the inspector
            Inspector.sense()
            Inspector.act()

            # Update the camera
            # cam_position = update_camera_position(Inspector.state[0:3])
            # CAM.position = cam_position
            # dc.client()
            # sleep(.1)

            # Test if docked
            distance_offset = Inspector.state[0:3] - Inspector.target_state[0:3]
            relative_velocity = Inspector.state[3:6] - Inspector.target_state[3:6]
            if all(distance_offset < close_enough) and all(relative_velocity < low_velocity):
                # print(Inspector.t)
                trial_output.append(Inspector.t)
                break

        if time_step == number_of_time_steps - 1:
            # print(np.nan)
            trial_output.append(np.nan)

    output[str(initial_state[0:3])] = pd.Series(trial_output)
df = pd.DataFrame.from_dict(output)
