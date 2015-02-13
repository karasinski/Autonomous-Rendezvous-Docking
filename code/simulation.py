from cv_tools import *
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


def RunSimulation(name):
    output = OrderedDict()
    for initial_state in initial_conditions:
        print(initial_state)

        # Run the simulation many times
        number_of_simulations = 100
        trial_output = []
        for _ in range(number_of_simulations):
            # Initialize Inspector

            if name is "Deliberative":
                Inspector = DeliberativeSatellite(initial_state, target_state, n)
            elif name is "Reactive":
                Inspector = ReactiveSatellite(initial_state, target_state, n)
            else:
                raise Exception

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

                # Try to use CV
                try:
                    # Scan the docking port
                    image, contours = scan_docking_port(target)

                    # Detect features
                    center, distance = detect_features(image, contours)

                    # Estimate state
                    state = estimate_state(center, distance, image)

                    print('Estimated: ', state)
                    print('   Actual: ', Inspector.state[0:3])
                    print('')
                except Exception:
                    print('State not found.')

                # Test if docked
                if Inspector.docked():
                    trial_output.append(Inspector.t)
                    break

            if time_step == number_of_time_steps - 1:
                trial_output.append(np.nan)

        output[str(initial_state[0:3])] = pd.Series(trial_output)
    df = pd.DataFrame.from_dict(output)
    return df

# Connect to EDGE
dc.connect()
dc.client()

# Find nodes
CAM = dc.Node("CM_Cam")
target = dc.Node("VR_PMA2_AXIAL_TARGET")

# Simulation parameters
n = 0.0011596575

# Inspector initial and final targets
initial_conditions = [[x, y, z, 0., 0., 0.] for x in range(100, 501, 100)
                                            for y in range(-50,  51,  50)
                                            for z in range(-50,  51,  50)]
target_state = [5., 0., 0., 0., 0., 0.]

Deliberative = RunSimulation("Deliberative")
Reactive = RunSimulation("Reactive")
results = {"Deliberative": Deliberative,
           "Reactive": Reactive}

d = pd.Panel(results)
