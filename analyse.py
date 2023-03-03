import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# Function to run c++ file for a set of parameters
def runSimulation(cppFile, params):
    '''
    Runs c++ simulation on input dictionary params and returns results as a matrix.
    Creates and then deletes two temporary files to pass parameters into and output data out of the c++ code.
    Results matrix contains output at each timestep as each row. Each variable is it's own column.

    Inputs:
        cppFile: String,    Name of c++ file
        params: dict,   Simulation Parameters

    Outputs:
        simulationData: np.array(nsteps+1, k),  matrix containing all output data across k variables
    '''

    # Set name of temporary output data file
    tempDataFileName = 'output.out'
    params['output'] = tempDataFileName

    # Create temporary file of parameters to get passed into simulation. Deleted after use.
    tempParamsFileName = 'configuration_temp.in'
    with open(tempParamsFileName, 'w') as temp:
        for key in params.keys():
            temp.write("%s %s %s\n" % (key, "=", params[key]))

    # Run simulation from terminal, passing in parameters via temporary file
    os.system(' .\\' + cppFile + ".exe .\\" + tempParamsFileName)

    # Save output data in simData matrix
    simulationData = np.loadtxt(tempDataFileName)

    # Delete temp files
    # os.remove(tempParamsFileName)
    # os.remove(tempDataFileName)

    return simulationData


parameters = {
'tfin' : 100.0,
'nsteps' : 2000,
'output' : 'output.out',
'sampling' : 2,
'N_part' : 1000,
'N_bins' : 30,
'v0' : -3.5,
'gamma' : 0.2,
'D' : 0.3, 
'vhb' : 5,
'vlb' : -5,
'vg_D' : -4,
'vd_D' : 4,
'sigma0':0.04,
'vc' : 3, 
'initial_distrib' : 'Gaussian',
}



def plot_distribution(params):
    # Run simulation
    simulationData = runSimulation("Exercice5_2023_student", params)

    # Extract velocity at each sample (assuming at index 0)
    vs = simulationData[:, 0]
    # Extract number of particles at each sample (assuming at index 1)
    Ns = simulationData[:, 1]

    plt.figure()
    plt.rcParams.update({'font.size': 15})
    plt.plot(vs, Ns)
    plt.title("Ex 4.3.a) Simulated vs Analytical Atmospheric densities")
    plt.xlabel("Height z [m]")
    plt.ylabel("Air density $\\rho_0$ [$kgm^{-3}$]")
    plt.legend()
    plt.show()

plot_distribution(parameters)


# # Ex 4.3.a)
# def plot_density(params):
#     # Run simulation
#     simulationData = runSimulation("ex4_atmos", params)

#     # Extract height at each sample (assuming at index 0)
#     zs = simulationData[:, 0]
#     # Extract density at each sample (assuming at index 1)
#     rhos_simulated = simulationData[:, 1]
#     # Get corresponding analytical densities 
#     rhos_analytical = analytical_density(zs)

#     plt.figure()
#     plt.rcParams.update({'font.size': 15})
#     plt.plot(zs, rhos_simulated, label='Simulated')
#     plt.plot(zs, rhos_analytical, label='Analytical')
#     plt.title("Ex 4.3.a) Simulated vs Analytical Atmospheric densities")
#     plt.xlabel("Height z [m]")
#     plt.ylabel("Air density $\\rho_0$ [$kgm^{-3}$]")
#     plt.legend()
#     plt.show()


# # plot_density(parameters_atmos)


# # Ex 4.3.b)
# def density_convergence(params):
#     z_0_analytical = (gamma / (gamma - 1)) * ((k_B * T_0) / (m * g))  # Analytical height of top of atmosphere

#     epsilons = np.logspace(0, 3, 5, base=10.0)  # Variations in starting altitudes
#     delta_zs = np.logspace(1, 3, 4, base=10.0)  # Variations in step size
#     errors = np.zeros((len(epsilons), len(delta_zs)))

#     plt.figure()

#     for i, epsilon in enumerate(epsilons):
#         z_start = z_0_analytical - epsilon  # Starting altitude of simulation, offset from true start of atmosphere by epsilon
#         rho_start = analytical_density(z_start)  # Analytically calculating density at starting altitude

#         # Prepare and run simulation
#         params['z_start'] = z_start
#         params['rho_start'] = rho_start
#         params['P_start'] = 0.0  # Set to zero because being ignored
#         params[
#             'rho_stop'] = -1  # kg/m^3,   stopping density of simulation (-ve so that z_stop condition is reached first)
#         params['z_stop'] = 0  # m,        stopping altitude of simulation
#         params['direction'] = 'Top'  # Making it simulate from the top down

#         for j, delta_z in enumerate(delta_zs):
#             # params['dz'] = -1 * delta_z  # m,        change in height with each step, -ve because integrating downwards
#             params['dz'] = delta_z  # m
#             simulationData = runSimulation("ex4_atmos", params)
#             rho_final = simulationData[-1, 1]  # Final simulated value of rho, assumed to be at index 1
#             errors[i, j] = np.absolute(rho_final - rho_0)

#         plt.plot(delta_zs, errors[i], label="$z_{start} = $" + str(z_start), alpha=0.2)

#     plt.title("Ex 4.3.b) Error convergence")
#     plt.xlabel("$\delta z$ [m]")
#     plt.ylabel("Error in $\\rho$ at $z=0$")
#     plt.legend()
#     plt.show()


# # density_convergence(parameters_atmos)

# ########################################################## Ex 4.4 ##########################################################

# '''
# NOTE
# !!APOLLO 13 SIMULATION - NO ATMPOSHERE!!

# I am assuming that the python-c++ interface works the same as with the previous exercises, so that the
# python code can pass a seperate file of parameters as input to the c++ file, and the c++ file will create
# an output file with all data values printed.

# Each row in the outputted data file should be layed out as such
#     [t  x   y   r   vx  vy  v]
# where r = sqrt(x^2 + y^2) and v = sqrt(vx^2 + vy^2) are calculated in the c++ code.

# The simulation should stop once the value of r at a given step is below the parameter r_stop. This final 
# value should not be included in the output, i.e. the final value for r in the output data file should be 
# the smallest value of r that is still larger that r_stop.

# I do not yet know what the best unit for time, I have left it blank so far. I also don't know what 
# appropriate values of dt and epsilon are, so the code below will need to be adapted to once decent values
# are found.

# Simulation Parameters:
#     - m_A           Mass of vessel
#     - m_T           Mass of Earth
#     - r_0           Starting distance of vessel from center of Earth
#     - v_0           Starting velocity magnitude of vessel
#     - dt            Time step size in simulation
#     - r_stop        Stopping distance of simulation
#     - algorithm     Type of algorithm (RK4_fixed or RK4_adaptive)
#     - epsilon       Time precision (used for RK4_adaptive, 0 for RK4_fixed)


# '''

# # Physical constants
# rho_0 = 0  # kg/m^3,       Density of atmosphere (0 => no atmosphere)
# m_A = 5800  # kg,           Mass of vessel
# m_T = 5.972e24  # kg,           Mass of Earth
# R_T = 6378.1e3  # m,            Radius of Earth
# G = 6.674e-11  # m^3/kgs^2,    Gravitational constant
# R_0 = 310000e3  # m,            Starting distance of vessel from center of Earth
# v_0 = 1.25e3  # m/s,          Starting velocity magnitude of vessel
# alpha = 10.56 * 2 * math.pi / 360


# # E



# # Ex 4.4.a)
# def plot_trajectory(params):
#     # params['dt'] = 10
#     simulationData = runSimulation('ex4_apollo', params)
#     t = simulationData[:, 0]
#     x = simulationData[:, 1]
#     y = simulationData[:, 2]
#     r = np.sqrt(np.power(x, 2) + np.power(y, 2))

#     plt.figure()
#     plt.rcParams.update({'font.size': 15})
#     plt.subplot(2, 1, 1)
#     # plt.axis('equal')
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.scatter(x, y, color='green', s=0.1)
#     plt.gca().add_patch(plt.Circle((0, 0), R_T, color='blue'))  # Draw Earth on plot as circle
#     plt.title("Trajectory of vessel")
#     plt.xlabel("x [m]")
#     plt.ylabel("y [m]")

#     plt.subplot(2, 1, 2)
#     plt.plot(t, r)
#     plt.title("Radial distance vs time")
#     # plt.hlines(R_T, min(t), t[np.where(np.isnan(r))[0][0]], ls='--', color=['grey'])
#     plt.hlines(R_T, min(t), max(t), ls='--', color='grey')
#     plt.xlabel("Time [s]")
#     plt.ylabel("Radial distance [m]")

#     plt.suptitle("Ex 4.4.a) Motion of Apollo 13")
#     plt.subplots_adjust(top=1.0, bottom=0.095, left=0.125, right=0.9, hspace=0.0, wspace=0.195)
#     plt.show()


# # plot_trajectory(parameters_apollo)


# def get_simulation_values(params):
#     # Return h_min and v_max

#     simulationData = runSimulation('ex4_apollo', params)
#     # r_final = math.sqrt(simulationData[-1, 1]**2+simulationData[-1, 2]**2)  # Final value of r returned by simulation

#     x_final, y_final = 0.0, 0.0
#     if (np.isnan(simulationData[:, 1]).any()):
#         x_final = simulationData[
#             np.where(np.isnan(simulationData[:, 1]))[0][0] - 1, 1]  # Needed to handle nan values returned by simulation
#     else:
#         x_final = simulationData[-1, 1]
#     if (np.isnan(simulationData[:, 2]).any()):
#         y_final = simulationData[
#             np.where(np.isnan(simulationData[:, 2]))[0][0] - 1, 2]  # Needed to handle nan values returned by simulation
#     else:
#         y_final = simulationData[-1, 2]

#     # r_final = math.sqrt(x_final**2 + y_final**2)
#     r_final = np.sqrt(np.square(x_final) + np.square(y_final))
#     h_final = r_final - R_T  # Altitude is difference between dist from center of Earth and radius of Earth
#     v = np.sqrt(np.square(simulationData[:, 3]) + np.square(simulationData[:, 4]))
#     v_max = max(v)  # Max value of the magnitude of v returned by simulation

#     return h_final, v_max


# def analytical_solution():
#     # TODO: Needs to be added
#     h_final = 0
#     v_final = np.sqrt((v_0 ** 2) - 2 * G * m_T * ((1 / R_0) - (1 / R_T)))

#     return h_final, v_final


# def convergence_analysis(params):
#     dts = np.logspace(-0.5, 3, num=30)  # Create range of dts to be simulated over
#     error_final_hs = np.zeros_like(dts)
#     error_max_vs = np.zeros_like(dts)

#     print(analytical_solution())

#     for i, dt in enumerate(dts):
#         params['dt'] = dt
#         h_final_simulated, v_max_simulated = get_simulation_values(params)
#         h_final_analytical, v_max_analytical = analytical_solution()

#         print("Completion: ", str(100 * i / len(dts)), "%")

#         error_final_hs[i] = np.absolute(h_final_analytical - h_final_simulated)
#         error_max_vs[i] = np.absolute(v_max_analytical - v_max_simulated)

#     # Plot both convergences on the same plot
#     fig = plt.figure()
#     plt.rcParams.update({'font.size': 15})
#     host = fig.add_subplot(111)
#     par1 = host.twinx()

#     host.set_xlim(dts[0], dts[-1])
#     host.set_xscale('log')
#     host.set_ylim(0.9 * min(error_final_hs), 1.1 * max(error_final_hs))
#     par1.set_ylim(0.9 * min(error_max_vs), 1.1 * max(error_max_vs))

#     host.set_xlabel("$\delta t$ [s]")
#     host.set_ylabel("Error in final altitude [m]")
#     par1.set_ylabel("Error in max velocity [m/s]")

#     p0, = host.plot(dts, error_final_hs, label='$h_{min}$', color='g')
#     p1, = par1.plot(dts, error_max_vs, label='$v_{max}$', color='b')
#     lns = [p0, p1]
#     host.legend(handles=lns, loc='best')

#     host.yaxis.label.set_color(p0.get_color())
#     host.tick_params(axis='y', colors=p0.get_color())
#     par1.yaxis.label.set_color(p1.get_color())
#     par1.tick_params(axis='y', colors=p1.get_color())

#     fig.tight_layout()
#     plt.title("Ex 4.4.a) Error Convergence")
#     plt.show()

#     fig = plt.figure()
#     plt.rcParams.update({'font.size': 15})
#     host = fig.add_subplot(111)
#     par1 = host.twinx()

#     host.set_xlim(dts[0], dts[-1])
#     host.set_xscale('log')
#     host.set_ylim(0.9 * min(error_final_hs), 1.1 * max(error_final_hs))
#     par1.set_ylim(0.9 * min(error_max_vs), 1.1 * max(error_max_vs))
#     host.set_yscale('log')
#     par1.set_yscale('log')

#     host.set_xlabel("$\delta t$ [s]")
#     host.set_ylabel("Error in final altitude [m]")
#     par1.set_ylabel("Error in max velocity [m/s]")

#     p0, = host.plot(dts, error_final_hs, label='$h_{min}$', color='g')
#     p1, = par1.plot(dts, error_max_vs, label='$v_{max}$', color='b')
#     lns = [p0, p1]
#     host.legend(handles=lns, loc='best')

#     host.yaxis.label.set_color(p0.get_color())
#     host.tick_params(axis='y', colors=p0.get_color())
#     par1.yaxis.label.set_color(p1.get_color())
#     par1.tick_params(axis='y', colors=p1.get_color())

#     fig.tight_layout()
#     plt.title("Ex 4.4.a) Error Convergence")
#     plt.show()


# # convergence_analysis(parameters_apollo)


# # Ex 4.4.b)

# def plot_time_steps(params):
#     simulationData = runSimulation('ex4_apollo', params)
#     ts = simulationData[:, 0]
#     dts = np.ediff1d(ts)  # Differences between successive elements in ts

#     plt.figure()
#     plt.plot(ts[1:], dts)
#     plt.title("Ex 4.4.b) Time step sizes over time")
#     plt.xlabel("Time [s]")
#     plt.ylabel("Time step [s]")
#     plt.show()


# parameters_apollo = {
#     'tFin': 259200,
#     'G': G,
#     'R_t': R_T,
#     'm_A': m_A,  # kg,           Mass of vessel
#     'm_t': m_T,  # kg,           Mass of Earth
#     'r_0': R_0,  # m,            Starting distance of vessel from center of Earth
#     'x0': R_0,
#     'y0': 0,
#     'vx0': - v_0 * math.cos(alpha),
#     'vy0': v_0 * math.sin(alpha),
#     'v_0': v_0,  # m/s,          Starting velocity magnitude of vessel
#     'dt': 10000,  # ...,          Time step size in simulation
#     'r_stop': R_T,  # m,            Stopping distance of simulation
#     'sampling': 1,
#     'g': g,
#     'Cx': 0.0,
#     'rho_0': 1.2,
#     'P_0': 1e5,
#     'gamma': 1.4,
#     'D': 4,
# }

# parameters_apollo['Algorithm'] = 'RK4TS'  # Type of algorithm (Adaptive-step Runge-Kutta 4th order)
# parameters_apollo['epsilon'] = 0.1


# #plot_time_steps(parameters_apollo)

# # Work In Progress
# # def compare_algorithms(params):
# #     nsteps = np.linspace(1e4, 1e6, 5)

# #     dts = 


# ########################################################## Ex 4.5 ##########################################################

# def get_max(Data):
#     a_max = max(Data[:, 5])
#     p_max = max(Data[:, 6])
#     return a_max, p_max


# def get_max_simulated(params):
#     data = runSimulation('ex4_apollo', params)
#     return get_max(data)


# def plot_trajectory_atm(params):
#     # params['dt'] = 10
#     simulationData = runSimulation('ex4_apollo', params)
#     t = simulationData[:, 0]
#     x = simulationData[:, 1]
#     y = simulationData[:, 2]
#     r = np.sqrt(np.power(x, 2) + np.power(y, 2))
#     vx = simulationData[:, 3]
#     vy = simulationData[:, 4]
#     acc = simulationData[:, 5]
#     power = simulationData[:, 6]

#     plt.figure()
#     plt.rcParams.update({'font.size': 15})
#     plt.subplot(2, 1, 1)
#     # plt.axis('equal')
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.scatter(x, y, color='green', s=0.1)
#     plt.gca().add_patch(plt.Circle((0, 0), R_T, color='blue'))  # Draw Earth on plot as circle
#     plt.title("Trajectory of vessel")
#     plt.xlabel("x [m]")
#     plt.ylabel("y [m]")

#     plt.subplot(2, 1, 2)
#     plt.plot(t, r)
#     plt.title("Radial distance vs time")
#     # plt.hlines(R_T, min(t), t[np.where(np.isnan(r))[0][0]], ls='--', color=['grey'])
#     plt.hlines(R_T, min(t), max(t), ls='--', color='grey')
#     plt.xlabel("Time [s]")
#     plt.ylabel("Radial distance [m]")

#     plt.suptitle("Ex 4.4.a) Motion of Apollo 13")
#     plt.subplots_adjust(top=1.0, bottom=0.095, left=0.125, right=0.9, hspace=0.0, wspace=0.195)
#     plt.show()

#     max_acceleration, max_power = get_max(simulationData)

#     print('Maximum acceleration during the simulation:')
#     print(max_acceleration)
#     print('Maximum power due to air resistance:')
#     print(max_power)

#     plt.figure()
#     plt.plot(t, acc)
#     plt.show()


# def convergence_power(params):
#     dts = np.logspace(-0.5, 3, num=30)  # Create range of dts to be simulated over
#     error_a = np.zeros_like(dts)
#     error_p = np.zeros_like(dts)

#     params['Algorithm'] = 'RK4TS'
#     params['dt'] = 1000
#     simulated_data = runSimulation('ex4_apollo', params)
#     a_ref, p_ref = get_max(simulated_data)
#     tFin_ref = simulated_data[-1, 0]

#     params['Algorithm'] = 'RK4'
#     for i, dt in enumerate(dts):
#         params['dt'] = dt

#         simulationData = runSimulation('ex4_apollo', params)
#         a_max, p_max = get_max(simulationData)

#         print("Completion: ", str(100 * i / len(dts)), "%")

#         error_a[i] = np.absolute(a_ref - a_max)
#         error_p[i] = np.absolute(p_ref - p_max)

#     plt.figure()
#     plt.plot(dts, error_a)
#     plt.xscale('log')
#     #plt.yscale('log')
#     plt.ylabel('Difference for max acceleration')
#     plt.xlabel('dt [s]')

#     plt.figure()
#     plt.plot(dts, error_p)
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.ylabel('Difference for max power')
#     plt.xlabel('dt [s]')

#     plt.show()


# parameters_apollo_atm = {
#     'tFin': 200000,  # 3 days = 259200
#     'G': G,
#     'g': g,
#     'R_t': R_T,
#     'm_A': m_A,  # kg,           Mass of vessel
#     'm_t': m_T,  # kg,           Mass of Earth
#     'r_0': R_0,  # m,            Starting distance of vessel from center of Earth
#     'x0': R_0,
#     'y0': 0,
#     'vx0': - v_0 * math.cos(alpha),
#     'vy0': v_0 * math.sin(alpha),
#     'v_0': v_0,  # m/s,          Starting velocity magnitude of vessel
#     'dt': 10000,  # ...,          Time step size in simulation
#     'r_stop': R_T,  # m,            Stopping distance of simulation
#     'epsilon': 0.1,
#     'sampling': 1,
#     'Algorithm': 'RK4TS',
#     'Cx': 0.3,
#     'rho_0': 1.2,
#     'P_0': 1e5,
#     'gamma': 1.4,
#     'D': 4,
# }

# #convergence_power(parameters_apollo_atm)

# parameters_apollo_atm['dt'] = 10000

# plot_trajectory_atm(parameters_apollo_atm)


# parameters_apollo_atm['tFin'] = 400000
# parameters_apollo_atm['Algorithm'] = 'RK4'
# parameters_apollo_atm['dt'] = 1000
# #plot_trajectory_atm(parameters_apollo_atm)

# parameters_apollo_atm['Algorithm'] = 'RK4'
# parameters_apollo_atm['dt'] = 100
# #plot_trajectory_atm(parameters_apollo_atm)

# parameters_apollo_atm['Algorithm'] = 'RK4'
# parameters_apollo_atm['dt'] = 10
# #plot_trajectory_atm(parameters_apollo_atm)

# def convergence_angle(params):
#     alpha_zero = 0 #10.562 * math.pi / 360
#     noise = np.random.normal(0, 0.15, 500)
#     alphas = alpha_zero + noise  # Create range of dts to be simulated over
#     # error_a = np.zeros_like(dts)
#     # error_p = np.zeros_like(dts)
#     a_max = np.zeros_like(alphas)
#     p_max = np.zeros_like(alphas)


#     params['Algorithm'] = 'RK4TS'
#     params['dt'] = 1000

#     for i, alpha in enumerate(alphas):
#         params['vx0'] = - v_0 * math.cos(alpha)
#         params['vy0'] = v_0 * math.sin(alpha)
#         simulationData = runSimulation('ex4_apollo', params)
#         a_max[i], p_max[i] = get_max(simulationData)
#         # if a_max[i] > 20:
#         #     a_max[i] = 0
#         #     p_max[i] = 0

#     plt.figure()
#     plt.scatter(alphas*360/(2*math.pi), a_max, s=0.8)
#     plt.ylabel('max acceleration')
#     plt.xlabel('alpha [°]')
#     plt.hlines(y=10., xmin=alpha_zero-40, xmax=alpha_zero+40, colors='red')

#     plt.show()

#     plt.figure()
#     plt.scatter(alphas*360/(2*math.pi), p_max, s=0.8)
#     plt.ylabel('max power')
#     plt.xlabel('alpha [°]')


#     plt.show()


# convergence_angle(parameters_apollo_atm)
