from __future__ import division
import numpy as np
import random
import simplejson
import matplotlib.pyplot as plt
from numpy import cos, sin, mat, array, vstack
from numpy.linalg import solve, norm


def ClohessyWiltshire(x0, y0, z0, dx0, dy0, dz0, t):
    '''Clohessy-Wiltshire equations'''

    x = (4-3*cos(n*t)) * x0 + (sin(n*t)/n)*dx0 + (2/n)*(1-cos(n*t))*dy0
    y = 6*(sin(n*t) - n*t)*x0 + y0 - (2/n)*(1-cos(n*t))*dx0 + 1/n * (4*sin(n*t) - 3*n*t)*dy0
    z = z0*cos(n*t) + dz0/n * sin(n*t)

    return [x, y, z, dx0, dy0, dz0, t]


def DeltaV(n, t, input_vec):
    '''find the delta_v required to take the system to zero for some time.
    assume you start with a known position (x0, y0, z0) '''

    x0 = input_vec[0]
    y0 = input_vec[1]
    z0 = input_vec[2]

    # solve Ax = b
    b = np.array([(3.*cos(n*t)-4.)*x0, y0 + 6.0*x0 *(sin(n*t) - n*t)])

    A = np.array([[(1./n)*sin(n*t), (2./n)*(1.-cos(n*t))],
                  [(2.0 / n) * (1.0 - cos(n * t)), 3.0 * t - (4.0 * sin(n * t) / n)]])

    x = solve(A,b)
    xd0 = x[0]
    yd0 = x[1]

    zd0 = -n * z0 * cos(n * t) / sin(n * t)

    burn_vec = input_vec
    burn_vec[3] = xd0
    burn_vec[4] = yd0
    burn_vec[5] = zd0

    return burn_vec

def ConductRendez(vec, t_burn_time, dt):
    '''perform the rendezvous using the DeltaV values attained above
    and the ClohessyWiltshire equations to compute the new position'''
    t = 0
    output = array(vec)

    while t< t_burn_time:
        temp = ClohessyWiltshire(vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], t)
        t+= dt

        vec[-1] = t
        output = np.vstack((output, temp))

    return output

def MultipleBurn(t0, t_burn_time, t_final, input_vec, dt, per1, per2):
    t = t0
    length_of_burn = t_burn_time
    num_of_min = t_final / length_of_burn
    burns = np.linspace(t, t_final - length_of_burn, num=num_of_min)

    #include the initial condition in the final result array
    result = array(input_vec)
    result = np.vstack((result, result)) #might need to remove this??

    for burn in burns:
        current_state = list(result[-1])
        t_fin = t_final - burn

        new_state = DeltaV(n, t_fin, current_state)

        #add 10% error to velocities
        new_state[3] = new_state[3] * random.uniform(per1,per2)
        new_state[4] = new_state[4] * random.uniform(per1,per2)
        new_state[5] = new_state[5] * random.uniform(per1,per2)

        # print new_state
        new_rendez = ConductRendez(new_state, length_of_burn, dt)
        result = np.vstack((result, np.array(new_rendez)))
    return result

def CalcTotalDeltaV(result_vec):
    ''' calculate the total delta_v burn in the x, y, z directions'''
    dx = result_vec[:,3]
    dy = result_vec[:,4]
    dz = result_vec[:,5]

    # grab the last burn and add to sum to account for final correction
    x_dv = np.abs(dx[-1])
    y_dv = np.abs(dy[-1])
    z_dv = np.abs(dz[-1])

    x_dv += np.sum(np.abs(np.diff(dx)))
    y_dv += np.sum(np.abs(np.diff(dy)))
    z_dv += np.sum(np.abs(np.diff(dx)))

    totalDv = x_dv + y_dv + z_dv

    return totalDv

def MultiRunSimulation(t0, t_final, t_burn_time, input_vec, dt):
    '''Conduct multiple simulations and find the fuel usage.
    Then create a histogram of the fuel usage and display the gaussian'''

    fuel_usage = []#np.array([0])
    percent_val1 = [0.95,0.9,0.85,0.8]
    percent_val2 = [1.05, 1.1, 1.15, 1.2]
    per_name = ["5%", "10%", "15%", "20%"]

    fuel_stats = []
    for j in range(len(percent_val1)):
        fuel_usage=[]
        per1 = percent_val1[j]
        per2 = percent_val2[j]
        name = per_name[j]

        for i in range(500):
            trial_run = MultipleBurn(t0, t_burn_time, t_final, input_vec, dt, per1, per2)
            usage = CalcTotalDeltaV(trial_run)
            usage = np.array(usage)

            fuel_usage.append(usage)

        filename = "TrialRun" + name +"Error.png"
        fuel_mean, fuel_stdev = PlotHist(fuel_usage, filename)

        fuel_usage.append(fuel_mean)
        fuel_usage.append(fuel_stdev)
        fuel_usage = np.array(fuel_usage)

        fuel_stats.append(fuel_usage)

    # fuel_usage = np.array(fuel_usage)
    fuel_stats = np.array(fuel_stats)
    print fuel_stats
    text_file = open('fuel_stats.txt', 'w')
    text_file.write("\n".join(str(elem) for elem in fuel_stats))
    # simplejson.dump(fuel_stats, text_file)
    text_file.close()

    return fuel_usage

def TimeVarySim(t0, t_burn_time, input_vec, dt):
    '''Conduct multiple simulations and find the fuel usage.
    Then create a histogram of the fuel usage and display the gaussian'''

    fuel_usage = []
    remaining_fuel = []
    total_fuel = 3.0
    per10_1 = 0.9
    per10_2 = 1.1
    # time_vec = np.linspace(30, 60.*60., 120) #simulate from 30s to 1 hour t_final
    time_vec = np.linspace(2600, 2800., 150) #simulate from 2600s t0 2800s every second

    for j in range(len(time_vec)):
        print j
        t_final = time_vec[j]
        name = str(round(time_vec[j]))

        trial_run = MultipleBurn(t0, t_burn_time, t_final, input_vec, dt, per10_1, per10_2)
        usage = CalcTotalDeltaV(trial_run)
        usage = np.array(usage)
        fuel_usage.append(usage)

        leftover = total_fuel - usage
        leftover = np.array(leftover)
        remaining_fuel.append(leftover)

        filename = "TrialRun" + name +"s_10%Error.png"
        Plot(trial_run, dt, filename)


    # print remaining_fuel
    fuel_usage = np.array(fuel_usage)
    remaining_fuel = np.array(remaining_fuel)

    PlotFuel(fuel_usage, remaining_fuel, time_vec)
    return fuel_usage

def Plot(results, dt, filename):
    '''Plot the position of the craft'''
    f, (ax1, ax2) = plt.subplots(2, 1)

    ax1.set_title('Position After Separation Using Multiple Burns')
    ax1.plot(results[:,0], label = 'x')
    ax1.plot(results[:,1], label = 'y')
    ax1.plot(results[:,2], label = 'z')
    # ax1.set_xlim([0.0, 80000])
    # ax1.set_ylim([-200., 1000.])
    ax1.set_ylabel("Position")
    ax1.set_xlabel("Time (s)")
    ax1.legend()

    ax2.set_title('Velocity After Separation Using Multiple Burns')
    ax2.plot(results[:,3], label = 'dx')
    ax2.plot(results[:,4], label = 'dy')
    ax2.plot(results[:,5], label = 'dz')
    # ax2.set_xlim([0.0, 80000])
    # ax2.set_ylim([-0.3, 0.4])
    ax2.set_ylabel("Velocity")
    ax2.set_xlabel("Time (s)")
    ax2.legend()

    plt.savefig(filename, format = "png")
    # plt.show()

def PlotHist(results, filename):
    '''Plot a histogram showing the fuel usage after multiple runs'''
    mu = np.mean(results)
    sigma = np.std(results)
    print mu, sigma

    count, bins, ignored = plt.hist(results, 20, normed = True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
                  np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
                 linewidth=2, color='r')
    plt.title('Gaussian Histogram for Fuel Usage')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    plt.savefig(filename, format = 'png')
    # plt.show()

    return mu, sigma

def PlotFuel(fuel, remaining, time):
    plt.figure()
    plt.title('Consumed and Remaining Fuel')
    plt.plot(time, fuel, label = 'consumed fuel')
    plt.plot(time, remaining, label= 'remaining fuel')
    plt.ylabel('Fuel (m/s)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.savefig("fuel.png", format = "png")
    plt.show()

# Simulation ##########################################################################33
n = 0.0011596575
dt = 0.05

# initial position
x0, y0, z0 = 100., 10., 10.

# initial velocity
dx0, dy0, dz0 = 0., 0., 0.

# initial time
t0 = 0.
t_final = 600.
t_burn_time = 60.

initial_cond= np.array([x0, y0, z0, dx0, dy0, dz0, t0])

# DeltaV(n, t_final, initial_cond)
# answer = ConductRendez(initial_cond, t_burn_time, dt)

#total_fuel = MultiRunSimulation(t0, t_final, t_burn_time, initial_cond, dt)
total_fuel = TimeVarySim(t0, t_burn_time, initial_cond, dt)
# PlotHist(total_fuel)

# answer2 = MultipleBurn(t0, t_burn_time, t_final, initial_cond, dt)

# CalcTotalDeltaV(answer2)
# Plot(answer2, dt)
