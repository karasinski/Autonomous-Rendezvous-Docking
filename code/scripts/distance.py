from cv_tools import *
import matplotlib
import pandas as pd
import numpy as np
from scipy.optimize import leastsq
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Connect to EDGE
dcomm.connect()
dcomm.client()

# Find nodes
CAM = dcomm.Node("CM_Cam")
target = dcomm.Node("VR_PMA2_AXIAL_TARGET")

# distances to test
distances = np.array(range(5, 101, 5), dtype=np.float64)

data = []
for x in distances:
    CAM.position = [0., 0., x]
    dcomm.client()

    try:
        # Scan the docking port
        image, contours = scan_docking_port(target)

        # Detect features
        radius = detect_features(image, contours)

        data = np.append(data, [x, radius])
    except Exception:
        pass

l = len(data)
data = data.reshape(l / 2, 2)

d = pd.DataFrame(data)
d.columns = ['distance', 'radius']


def fitfunc(p, x):
    y = p[0] * np.exp(p[1] * x) + p[2] * x ** 2 + p[3] * x + p[4]
    return y


def residuals(p, x, y):
    return y - fitfunc(p, x)


x = np.array(d.radius)
y = np.array(d.distance)

p_guess = [2.10044493e+02, -3.68823114e-01, 4.66638416e-02, -3.08006328e+00, 5.54876863e+01]
p, cov, infodict, mesg, ier = leastsq(residuals, p_guess, args=(x, y), full_output=True)
xp = np.linspace(0, 35)
pxp = fitfunc(p, xp)
print(p)

f, (ax1, ax2) = plt.subplots(1, 2)

# Plot fit vs actual data
ax1.plot(x, y, 'ko', label='Original Data')
ax1.plot(xp, pxp, 'r-', label='Fitted Curve')
plt.xlim(0, 35)
plt.legend()

# Plot RMS
resid = residuals(p, x, y)
ax2.plot(x, resid, '.', label='RMS')
plt.xlim(0, 35)
plt.legend()
plt.show()
