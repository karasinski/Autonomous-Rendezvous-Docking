import matplotlib.pyplot as plt
import numpy as np
import cv2
import dcomm as dc
from time import sleep
import urllib
import math


def write_ply(fn, verts, colors):
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''

    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')


def load_image():
    width = 1600
    height = 900
    try:
        # Give the server some room to breath
        sleep(.1)

        req = urllib.urlopen(
            'http://127.0.0.1:8080/image?width=' + str(width) + '&height=' + str(height))
    except IOError:
        print('EDGE is not running as a local server.')

    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)  # 'load it as it is'
    return img


# Connect to EDGE
dc.connect()
dc.client()

# Find nodes
CAM = dc.Node("CM_Cam")

position = np.array([0., 50., 50.])

CAM.position = position - np.array([0., 5., 0.])
dc.client()
right = load_image()
cv2.imwrite('right.png', right)

CAM.position = position + np.array([0., 5., 0.])
dc.client()
left = load_image()
cv2.imwrite('left.png', left)

print 'loading images...'
# downscale images for faster processing
imgL = cv2.imread('left.png')
imgR = cv2.imread('right.png')
# disparity range is tuned for 'aloe' image pair
window_size = 3
min_disp = 16
num_disp = 112 - min_disp
stereo = cv2.StereoSGBM(minDisparity=min_disp,
                        numDisparities=num_disp,
                        SADWindowSize=window_size,
                        uniquenessRatio=10,
                        speckleWindowSize=100,
                        speckleRange=32,
                        disp12MaxDiff=1,
                        P1=8 * 3 * window_size**2,
                        P2=32 * 3 * window_size**2,
                        fullDP=True
                        )
print 'computing disparity...'
disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
print 'generating 3d point cloud...',
h, w = imgL.shape[:2]
f = 0.8 * w  # guess for focal length
Q = np.float32([[1, 0, 0, -0.5 * w],
                [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
                [0, 0, 0, -f],  # so that y-axis looks up
                [0, 0, 1, 0]])
points = cv2.reprojectImageTo3D(disp, Q)
colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
mask = disp > disp.min()
out_points = points[mask]
out_colors = colors[mask]
out_fn = 'out.ply'
write_ply('out.ply', out_points, out_colors)
print '%s saved' % 'out.ply'
# cv2.imshow('left', imgL)
disparity = (disp - min_disp) / num_disp
# cv2.imshow('disparity', disparity)
# cv2.imwrite('disparity.png', disparity)
# cv2.waitKey()
# cv2.destroyAllWindows()
plt.imshow(disparity)

dist = []
for i in range(points.shape[0]):
    for j in range(points.shape[1]):
        ndist = math.sqrt(sum(points[i][j]**2))
        dist.append(ndist)
        # dist = np.vstack((dist, ndist))
        # print dist 

plt.hist(dist, bins = 100)
plt.show()