from __future__ import print_function, division
import dcomm as dc
import numpy as np
import pandas as pd
import cv2
from time import sleep
import urllib


def get_image():
    width, height = 800, 900
    try:
        # Give the server some room to breath
        sleep(.05)

        req = urllib.urlopen(
            'http://127.0.0.1:8080/image?width=' + str(width) + '&height=' + str(height))
    except IOError:
        print('EDGE is not running as a local server.')

    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)  # 'load it as it is'
    return img


def load_data(file):
    d = pd.DataFrame.from_csv(file)
    d.columns = ['rx', 'ry', 'rz', 'dx', 'dy', 'dz']
    d = d.as_matrix()

    return d


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

# Find nodes
CAM = dc.Node("CM_Cam")

# Load data
d = load_data('laserdata')

t = 0
for row in d:
    t += 1
    print(t)
    cam_position = update_camera_position(row[0:3])
    CAM.position = cam_position
    dc.client()
    r = get_image()

    cam_position = update_camera_position(row[3:6])
    CAM.position = cam_position
    dc.client()
    d = get_image()

    rd = np.hstack((r, d))
    cv2.imwrite('movie/fin-' + format(t, '04d') + '.png', rd)


# ffmpeg -framerate 20 -i fin-%04d.png out2.mp4
