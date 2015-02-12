import matplotlib
import pandas as pd
import numpy as np
from scipy.optimize import leastsq
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


d = [
    [5, 29.552000999450684, 0.15203570251174794],
    [10, 21.974592208862305, 0.20147586240901158],
    [15, 18.020273208618164, 0.15533600598304825],
    [20, 14.75779104232788, 0.14961478466752048],
    [25, 12.339763641357422, 0.18965008933596814],
    [30, 10.786501407623291, 0.17148053192776219],
    [35, 9.604451179504395, 0.41159082031020905],
    [40, 8.611880874633789, 0.41967344420326835],
    [45, 7.596048831939697, 0.12306773746301523],
    [50, 6.966182351112366, 0.10410091844055704],
    [60, 5.936443448066711, 0.13907385631828154],
    [80, 4.295474410057068, 0.1025552369444663],
    [100, 3.3682428002357483, 0.17492293263668218]
    # [150, 1.828736573457718, 0.23423177790521874]
]


d = pd.DataFrame(d)
d.columns = ['distance', 'radius', 'error']


def fitfunc(p, x):
    y = p[0] * np.exp(p[1] * x) + p[2] * x ** 2 + p[3] * x + p[4]
    return y


def residuals(p, x, y):
    return y - fitfunc(p, x)


x = np.array(d.radius)
y = np.array(d.distance)

p_guess = [1.87599710e+02, -3.59449427e-01, 4.50895146e-02, -2.95463925e+00, 5.29948831e+01]
p, cov, infodict, mesg, ier = leastsq(residuals, p_guess, args=(x, y), full_output=True)
xp = np.linspace(0, 35)
pxp = fitfunc(p, xp)

# Plot fit vs actual data
plt.plot(x, y, 'ko', label='Original Data')
plt.plot(xp, pxp, 'r-', label='Fitted Curve')
plt.xlim(0, 35)
plt.legend()
plt.show()

# Plot RMS
plt.clf()
resid = residuals(p, x, y)
plt.plot(x, resid, '.', label='RMS')
plt.xlim(0, 35)
plt.legend()
plt.show()

# Offset data
d = [[ 20., 297.24105835],
     [ 30., 197.75      ],
     [ 40., 146.25      ],
     [ 50., 116.25      ],
     [ 60.,  94.75003052],
     [ 70.,  80.25      ],
     [ 80.,  69.40063477],
     [ 90.,  61.50006104],
     [100.,  56.25      ],
     [110.,  50.5       ],
     [120.,  46.37931824],
     [130.,  42.75      ],
     [140.,  40.        ],
     [150.,  36.88705444]]

d = pd.DataFrame(d)
d.columns = ['distance', 'pixels']
d['offset'] = 10.
d['times'] = d.distance * d.pixels / d.offset

# Distance * # Pixels ~ Constant * Offset