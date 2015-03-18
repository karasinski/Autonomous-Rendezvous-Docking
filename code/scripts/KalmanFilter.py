import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def KalmanFilter(input_signal, Q=None, R=None):
    # Default values for Q and R
    if Q is None:
        Q = 1e-4  # Process variance
    if R is None:
        R = 1e-3  # Estimate of measurement variance

    # Intial parameters
    sz = input_signal.shape

    # Allocate space for arrays
    xhat = np.zeros(sz)       # a posteri estimate of x
    P = np.zeros(sz)          # a posteri error estimate
    xhatminus = np.zeros(sz)  # a priori estimate of x
    Pminus = np.zeros(sz)     # a priori error estimate
    K = np.zeros(sz)          # gain or blending factor

    # Intial guesses
    xhat[0] = input_signal[0]
    P[0] = .0003

    for k in range(1, len(input_signal)):
        # Time update
        xhatminus[k] = xhat[k - 1]
        Pminus[k] = P[k - 1] + Q

        # Measurement update
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (input_signal[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]

    return xhat

# Read data and format dataframe
d = pd.read_table('data/data', header=None, sep=r"\s+")
d.columns = ['x', 'y', 'z', 'cvx', 'cvy', 'cvz']
d['r'] = d.x + d.y + d.z
d['cvr'] = d.cvx + d.cvy + d.cvz
d = d[['r', 'x', 'y', 'z', 'cvr', 'cvx', 'cvy', 'cvz']]

# Estimated value for x is x + cv error in x
est_x = np.array(d.x + d.cvx)

# Filter the signal
xhat = KalmanFilter(est_x)

# Plot results
plt.figure()
plt.plot(est_x, 'k+', label='Measurement')
plt.plot(xhat, 'b-', label='Estimate')
plt.plot(d.x, color='g', label='Truth')
plt.legend()
plt.xlabel('Timestep')
plt.ylabel('Distance')
plt.xlim([0, 1500])
plt.show()
