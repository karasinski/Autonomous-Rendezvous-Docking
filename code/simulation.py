from reactive import *
import dcomm as dc
from time import sleep


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

# Simulation parameters, initialize Inspector
n = 0.0011596575
initial_state = [100., 15., -55., 0., 0., 0.]
target_state = [10., 0., 0., 0., 0., 0.]
Inspector = Satellite(initial_state, target_state, n)

# Set the camera position to be equal to the Inspector position, sync with server
for _ in range(5000):
    state = Inspector.state[0:3]
    print(state)

    Inspector.sense()
    Inspector.act()

    cam_position = update_camera_position(Inspector.state[0:3])
    CAM.position = cam_position
    dc.client()
    sleep(.1)
